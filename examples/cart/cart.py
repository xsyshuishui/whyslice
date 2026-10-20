from pricing import discount_for


class Cart:
    def __init__(self, member=False):
        self.items = []
        self.member = member

    def add(self, name, price, qty=1):
        self.items.append((name, price, qty))

    def subtotal(self):
        total = 0
        for _, price, qty in self.items:
            total += price * qty
        return total

    def total(self):
        sub = self.subtotal()
        disc = discount_for(sub, self.member)
        return round(sub - disc, 2)
