from cart import Cart


def test_member_at_tier_threshold():
    c = Cart(member=True)
    c.add("book", 50, 2)
    c.add("pen", 25, 4)
    assert c.total() == 160.0
