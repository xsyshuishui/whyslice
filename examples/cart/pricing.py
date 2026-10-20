TIERS = [(100, 0.10), (200, 0.15), (500, 0.20)]


def tier_rate(amount):
    rate = 0.0
    for threshold, r in TIERS:
        if amount > threshold:
            rate = r
    return rate


def discount_for(subtotal, member):
    rate = tier_rate(subtotal)
    if member:
        rate += 0.05
    return round(subtotal * rate, 2)
