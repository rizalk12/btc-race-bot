"""Dip-bounce bot for agent-fun/bars-replay/v1 (make.fun BTC race).

Strategy: mean-reversion dip buying on 1-minute cbBTC bars.
  - A reference mean of the last REF_N (600) closes tracks the slow fair
    price; shallow dips under shorter means do not bounce in a grinding
    market, deep discounts under the 10-hour mean are the ones that did.
  - When flat, buy once the close sits at least DIP_BPS below that mean AND
    the fall has stalled (the lowest close of the recent bars is no longer
    falling and the last close ticks up), so the bot does not catch a knife
    in the middle of a plunge.
  - When holding, sell everything on the first of: take-profit versus the
    position's average cost, recovery back toward the reference mean, a
    stop-loss versus cost, or a maximum holding time.
  - After any exit, sit out a cooldown so noise cannot churn fees.
  - If equity falls far below its peak, stop opening new positions.
  - A window that never offers a deep dip must not end tradeless (the
    race bar asks for at least one trade): late in the window the dip
    requirement relaxes, and near the end a small starter position is
    opened on any bar that is not falling. Both fire only when no
    position was ever held, so a normal window never sees them.

The bot speaks the bars-replay wire protocol itself: newline-delimited JSON
on stdin/stdout, hello once, one batch per minute, exactly one reply per
batch echoing its seq, then done. Orders decided on a batch fill at the
next bar's open with a 4 bps fee, 2 bps of modelled slippage and
constant-product price impact, so a round trip has to clear roughly
25-35 bps before it earns anything; the thresholds below respect that.

Everything is integer arithmetic on the protocol's scaled integers, a pure
function of the lines received: no clock, no files, no network, no random,
so two runs on the same bars answer byte for byte the same.
"""

import json
import sys

SYMBOL = "CBBTC"

# --- strategy parameters (tuned on the published dev window) -------------
REF_N = 600          # bars in the reference mean (10 hours)
DIP_BPS = 100        # buy when the close is this far below the mean
STALL_LOOKBACK = 10  # lows must stop falling over this many bars
TP_BPS = 80          # take profit versus average cost
MEAN_EXIT_BPS = 25   # also exit once the close recovers toward the mean
SL_BPS = 120         # stop loss versus average cost
MAX_HOLD = 420       # bars; time stop
COOLDOWN = 30        # bars sat out after any exit
SIZE_BPS = 9000      # spend this share of equity per entry (90%)
LATE_SEQ = 1000      # if still tradeless here, accept shallower dips
LATE_DIP_BPS = 45    # ...this far below the mean instead
LAST_SEQ = 1320      # if still tradeless here, open a small starter
LAST_SIZE_BPS = 1500  # ...of this size, so the run registers a trade
DD_STOP_BPS = 1000   # no new entries once equity is 10% under its peak
BUY_SLIP_BPS = 250   # max_slippage_bps sent with buys
SELL_SLIP_BPS = 250  # max_slippage_bps sent with sells


def bps_of(num, den):
    """(num/den - 1) in basis points, integer, truncated toward zero."""
    q = (num * 10000) // den - 10000
    return q


class Bot:
    def __init__(self, hello):
        self.hello = hello
        self.sym = hello["symbols"][0]["symbol"]
        info = hello["symbols"][0]
        self.price_scale = info["price_scale"]
        self.qty_scale = info["qty_scale"]
        self.quote_scale = hello["capital"]["quote_scale"]
        self.min_notional = int(hello["fees"]["min_notional_quote_units"])
        self.warmup = hello.get("warmup_bars", 0)
        self.closes = []          # every close seen, oldest first
        self.cash = int(hello["capital"]["quote_units"])
        self.equity = self.cash
        self.peak_equity = self.equity
        self.qty = 0              # held quantity of the symbol
        self.cost = 0             # cost_quote_units of the held quantity
        self.ever_held = False    # a position was seen at least once
        self.entry_seq = -1       # batch seq at which the position appeared
        self.cooldown_until = -1  # no entries at seq <= this
        self.order_n = 0

    # -- account -----------------------------------------------------------
    def apply_account(self, account, seq):
        self.cash = int(account["cash_quote_units"])
        self.equity = int(account["equity_quote_units"])
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        qty = 0
        cost = 0
        for p in account["positions"]:
            if p["symbol"] == self.sym:
                qty = int(p["qty"])
                cost = int(p["cost_quote_units"])
        if qty > 0 and self.qty == 0:
            self.entry_seq = seq
            self.ever_held = True
        if qty == 0 and self.qty > 0:
            self.cooldown_until = seq + COOLDOWN
            self.entry_seq = -1
        self.qty = qty
        self.cost = cost

    def entry_price(self):
        """Average price paid per unit, at price_scale, fees included."""
        # value(qty, price) == cost  =>  price = cost * 10^(qs+ps) / (qty * 10^quote)
        return (self.cost * 10 ** (self.qty_scale + self.price_scale)) // (
            self.qty * 10 ** self.quote_scale
        )

    # -- orders --------------------------------------------------------------
    def client_id(self):
        self.order_n += 1
        return "o%d" % self.order_n

    def buy_order(self, notional):
        return {
            "side": "buy",
            "client_id": self.client_id(),
            "symbol": self.sym,
            "notional_quote_units": str(notional),
            "max_slippage_bps": BUY_SLIP_BPS,
        }

    def sell_all_order(self):
        return {
            "side": "sell",
            "client_id": self.client_id(),
            "symbol": self.sym,
            "qty": str(self.qty),
            "max_slippage_bps": SELL_SLIP_BPS,
        }

    # -- decision --------------------------------------------------------------
    def decide(self, seq):
        closes = self.closes
        if len(closes) < REF_N:
            return []
        close = closes[-1]
        ref = sum(closes[-REF_N:]) // REF_N
        dev = bps_of(close, ref)

        if self.qty > 0:
            held_for = seq - self.entry_seq if self.entry_seq >= 0 else 0
            pnl = bps_of(close, self.entry_price())
            if (
                pnl >= TP_BPS
                or dev >= MEAN_EXIT_BPS
                or pnl <= -SL_BPS
                or held_for >= MAX_HOLD
            ):
                return [self.sell_all_order()]
            return []

        # Flat: look for a deep, stalled dip under the long mean. The long
        # mean is only trusted once REF_N bars exist, so the bot sits out
        # the first REF_N bars of any window rather than guess.
        if seq <= self.cooldown_until:
            return []
        if self.warmup and seq < self.warmup:
            return []
        if bps_of(self.equity, self.peak_equity) <= -DD_STOP_BPS:
            return []
        # A window that never offers a deep dip must not end tradeless:
        # late in the window, accept a shallower dip, and near the very
        # end open a small starter on any bar that is not falling. These
        # only fire when no position was ever held, so a window with a
        # real dip (the normal case) never sees them.
        dip_needed = DIP_BPS
        if not self.ever_held and seq >= LATE_SEQ:
            dip_needed = LATE_DIP_BPS
        if not self.ever_held and seq >= LAST_SEQ and close >= closes[-2]:
            notional = self.equity * LAST_SIZE_BPS // 10000
            if notional > self.cash:
                notional = self.cash
            if notional >= self.min_notional:
                return [self.buy_order(notional)]
            return []
        if dev > -dip_needed:
            return []
        # The last bar must tick up: still falling is not a dip, it is a
        # plunge in progress.
        if close <= closes[-2]:
            return []
        # The fall must have stalled: the lowest close of the last
        # STALL_LOOKBACK bars is not below the lowest of the STALL_LOOKBACK
        # before them, and the last close is not a fresh low either.
        recent = closes[-STALL_LOOKBACK:]
        prior = closes[-2 * STALL_LOOKBACK:-STALL_LOOKBACK]
        if min(recent) < min(prior):
            return []
        if close <= min(recent[:-1]):
            return []
        notional = self.equity * SIZE_BPS // 10000
        if notional > self.cash:
            notional = self.cash
        if notional < self.min_notional:
            return []
        return [self.buy_order(notional)]


def main():
    bot = None
    out = sys.stdout
    for raw in sys.stdin.buffer:
        line = raw.strip()
        if not line:
            continue
        msg = json.loads(line.decode("utf-8"))
        kind = msg.get("type")
        if kind == "hello":
            bot = Bot(msg)
        elif kind == "batch":
            seq = msg["seq"]
            for bar in msg["bars"]:
                if bar["symbol"] == bot.sym:
                    bot.closes.append(int(bar["c"]))
            bot.apply_account(msg["account"], seq)
            orders = bot.decide(seq)
            reply = {"type": "reply", "seq": seq, "orders": orders}
            out.write(json.dumps(reply, separators=(",", ":")) + "\n")
            out.flush()
        elif kind == "done":
            bot.apply_account(msg["account"], msg["seq"])
            return
    # Input ended without done: exit quietly rather than hang.


if __name__ == "__main__":
    main()
