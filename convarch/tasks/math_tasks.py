"""Multi-step word problems with one numeric answer (GSM8K-style, written fresh).

`ablations` maps a shard id to the answer you get if that shard is ignored; the
validator checks that each differs from the gold answer.
"""

from .schema import Shard, Task

CONTEXT = "You are a helpful assistant that helps people work out practical calculations."

DELIVERABLE = (
    "Deliverable format: whenever you give a numeric result, end your message with a final "
    "line of the form `ANSWER: <number>` (digits only, no units or currency symbols)."
)

TASKS = [
    Task(
        id="math_catering",
        domain="math",
        title="Offsite catering bill",
        shards=[
            Shard(1, "what will catering for our team offsite cost in total?"),
            Shard(2, "40 people are coming."),
            Shard(3, "Lunch is $22 per person."),
            Shard(4, "A quarter of the group is vegetarian, and the vegetarian meal costs $3 more than the regular one."),
            Shard(5, "There's an 18% service charge on the food total."),
            Shard(6, "Plus a flat $75 delivery fee, which has no service charge on it."),
            Shard(7, "Update: 48 people are coming now.", revises=2, full_text="48 people are coming."),
            Shard(8, "We get a 5% nonprofit discount on the food, applied before the service charge."),
            Shard(9, "The offsite is in Napa.", distractor=True),
        ],
        gold=1299.132,
        ablations={4: 1258.776, 5: 1112.40, 6: 1224.132, 7: 1095.11, 8: 1363.56},
    ),
    Task(
        id="math_paint",
        domain="math",
        title="Paint cans to buy",
        shards=[
            Shard(1, "how many cans of paint do I need to buy for my bedroom?"),
            Shard(2, "The room is 12 feet by 10 feet, and the walls are 8 feet high."),
            Shard(3, "I'm not painting the door (3 by 7 feet) or the two windows (each 4 by 5 feet)."),
            Shard(4, "I want two coats."),
            Shard(5, "One can covers 150 square feet, and they only sell whole cans."),
            Shard(6, "Actually one coat is enough, it's the same color as now.", revises=4,
                  full_text="One coat is enough."),
            Shard(7, "I also want to paint the ceiling, same number of coats."),
            Shard(8, "I already have one full can at home."),
            Shard(9, "The paint is a pale sage green.", distractor=True),
        ],
        gold=2,
        ablations={3: 3, 5: 1.74, 6: 5, 7: 1, 8: 3},
    ),
    Task(
        id="math_commute",
        domain="math",
        title="Train vs. driving savings",
        shards=[
            Shard(1, "how much would I save per year taking the train to work instead of driving?"),
            Shard(2, "I work 5 days a week, 48 weeks a year."),
            Shard(3, "My round trip is 32 miles, my car gets 28 miles per gallon, and gas is $4.20 a gallon."),
            Shard(4, "Parking at the office is $15 a day."),
            Shard(5, "A monthly train pass is $210, and I'd buy one for 11 months of the year. The station is a short walk from home, so getting there costs nothing."),
            Shard(6, "I'd still drive in one day a week, so the train would only replace 4 of my 5 days. I'd still need the monthly pass, though."),
            Shard(7, "Gas just went up to $4.90 a gallon, use that.", revises=3,
                  full_text="My round trip is 32 miles, my car gets 28 miles per gallon, and gas is $4.90 a gallon."),
            Shard(8, "My employer reimburses half the cost of the train pass."),
            Shard(9, "I live out in the suburbs.", distractor=True),
        ],
        gold=2800.20,
        ablations={4: -79.80, 5: 3955.20, 6: 3789.00, 7: 2646.60, 8: 1645.20},
    ),
    Task(
        id="math_pizza",
        domain="math",
        title="Birthday pizza order",
        shards=[
            Shard(1, "how much will the pizza for my son's birthday party cost?"),
            Shard(2, "There will be 12 kids."),
            Shard(3, "Each kid eats about 3 slices, and the adults will eat 10 slices in total."),
            Shard(4, "A large pizza has 8 slices and costs $16.50."),
            Shard(5, "We can only order whole pizzas."),
            Shard(6, "There's a coupon for 20% off the whole order."),
            Shard(7, "Update: 15 kids have confirmed now, not 12.", revises=2,
                  full_text="There will be 15 kids."),
            Shard(8, "Order 2 extra pizzas on top of that, just in case."),
            Shard(9, "Oh wait, the coupon is actually $20 off the whole order, not a percentage.", revises=6,
                  full_text="There's a coupon for $20 off the whole order."),
            Shard(10, "The party's at the park.", distractor=True),
        ],
        gold=128.50,
        ablations={5: 126.4375, 7: 112.00, 8: 95.50, 9: 118.80},
    ),
    Task(
        id="math_down_payment",
        domain="math",
        title="Months to save a down payment",
        shards=[
            Shard(1, "how many months until I've saved enough for a house down payment?"),
            Shard(2, "The house costs $450,000 and I need 20% down."),
            Shard(3, "I already have $38,000 saved."),
            Shard(4, "I can put away $2,500 at the end of every month."),
            Shard(5, "My parents are giving me $10,000 toward it right at the start."),
            Shard(6, "I also want to keep an extra $5,000 on top of the down payment as an emergency cushion."),
            Shard(7, "Actually the house I like now is $480,000.", revises=2,
                  full_text="The house costs $480,000 and I need 20% down."),
            Shard(8, "I'll also get a $6,000 bonus at the end of month 6 that goes straight into savings."),
            Shard(9, "I'm looking in Sacramento.", distractor=True),
        ],
        gold=19,
        ablations={3: 34, 5: 23, 6: 17, 7: 17, 8: 22},
    ),
    Task(
        id="math_road_trip",
        domain="math",
        title="Road-trip cost split",
        shards=[
            Shard(1, "how much does each of my friends owe me for the road trip I paid for?"),
            Shard(2, "There were 4 of us including me, splitting the costs equally."),
            Shard(3, "We drove 1,260 miles total, the car averages 30 miles per gallon, and gas averaged $3.80 a gallon."),
            Shard(4, "The hotel was 3 nights at $189 per night, plus 12% tax."),
            Shard(5, "Tolls came to $46."),
            Shard(6, "I'm covering the hotel tax myself as a thank-you, so don't split that part."),
            Shard(7, "We also paid $84 for parking at the hotel, split equally like everything else."),
            Shard(8, "Oh, one friend dropped out before the trip, so it was 3 of us including me.", revises=2,
                  full_text="There were 3 of us including me, splitting the costs equally."),
            Shard(9, "We went to Yosemite.", distractor=True),
        ],
        gold=285.5333,
        ablations={3: 232.3333, 5: 270.20, 6: 308.2133, 7: 257.5333, 8: 214.15},
    ),
]
