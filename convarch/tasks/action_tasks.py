"""Function-calling tasks: the deliverable is exactly one API call.

The API is described by `API`. Grading compares the call with the gold call
argument by argument: the type of each parameter sets how values are normalized,
optional arguments that are left out take their default, and unknown arguments
fail the call. Free-text arguments (titles) only need to be non-empty.
"""

import json

from .schema import Shard, Task

TODAY = "Wednesday, March 11, 2026"

CONTACTS = {
    "Priya Raman": "priya.raman@acme.io",
    "Marco Bianchi": "marco.bianchi@acme.io",
    "Dana Wu": "dana.wu@acme.io",
    "Sam Okafor": "sam.okafor@acme.io",
    "Lena Fischer": "lena.fischer@acme.io",
}

DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

# param -> (type, required, default, description)
API = {
    "create_event": {
        "description": "Create a calendar event on the user's calendar.",
        "params": {
            "title": ("free", True, None, "Event title."),
            "start": ("datetime", True, None, "Local start time, 'YYYY-MM-DDTHH:MM'."),
            "duration_minutes": ("int", True, None, "Length of the event in minutes."),
            "attendees": ("set", False, [], "List of attendee email addresses (not including the user)."),
            "location": ("text", False, None, "Room or place."),
            "reminder_minutes": ("int", False, 10, "Minutes before start to remind the user."),
            "recurrence": ("enum:none,daily,weekly,monthly", False, "none", "Repeat rule."),
            "video_call": ("bool", False, False, "Attach a video-call link."),
        },
    },
    "search_flights": {
        "description": "Search for flights.",
        "params": {
            "origin": ("code", True, None, "Origin airport IATA code."),
            "destination": ("code", True, None, "Destination airport IATA code."),
            "depart_date": ("date", True, None, "'YYYY-MM-DD'."),
            "return_date": ("date", False, None, "'YYYY-MM-DD' for round trips; null for one-way."),
            "passengers": ("int", False, 1, "Number of passengers."),
            "cabin": ("enum:economy,premium_economy,business,first", False, "economy", "Cabin class."),
            "max_stops": ("int", False, None, "Maximum number of stops; null for any."),
            "sort_by": ("enum:price,duration,departure_time", False, "price", "Result ordering."),
        },
    },
    "book_table": {
        "description": "Reserve a restaurant table.",
        "params": {
            "restaurant": ("text", True, None, "Restaurant name."),
            "date": ("date", True, None, "'YYYY-MM-DD'."),
            "time": ("time", True, None, "24-hour 'HH:MM'."),
            "party_size": ("int", True, None, "Number of guests including the user."),
            "seating": ("enum:any,indoor,outdoor,bar", False, "any", "Seating preference."),
            "occasion": ("enum:none,birthday,anniversary,business", False, "none", "Special occasion."),
            "dietary_restrictions": ("set:vegetarian,vegan,gluten_free,nut_allergy,dairy_free", False, [],
                                     "Dietary needs in the party."),
        },
    },
    "create_ticket": {
        "description": "File an issue in the team's tracker.",
        "params": {
            "project": ("enum:WEB,MOBILE,INFRA,DATA", True, None, "Project key."),
            "title": ("free", True, None, "Short summary."),
            "type": ("enum:bug,task,story", True, None, "Issue type."),
            "priority": ("enum:low,medium,high,urgent", False, "medium", "Priority."),
            "assignee": ("code", False, None, "Assignee email; null for unassigned."),
            "labels": ("set", False, [], "Labels."),
            "due_date": ("date", False, None, "'YYYY-MM-DD'."),
        },
    },
    "set_thermostat_schedule": {
        "description": "Add a recurring schedule to the smart thermostat.",
        "params": {
            "zone": ("enum:living_room,bedroom,office,whole_house", True, None, "Zone."),
            "days": (f"set:{','.join(DAYS)}", True, None, "Days the schedule applies to."),
            "start_time": ("time", True, None, "24-hour 'HH:MM'."),
            "end_time": ("time", True, None, "24-hour 'HH:MM' (may be on the next day)."),
            "target_temp": ("number", True, None, "Target temperature."),
            "unit": ("enum:F,C", False, "F", "Temperature unit."),
            "mode": ("enum:heat,cool,auto", False, "auto", "HVAC mode."),
        },
    },
    "rent_car": {
        "description": "Reserve a rental car.",
        "params": {
            "pickup_location": ("code", True, None, "Airport IATA code or city name."),
            "pickup": ("datetime", True, None, "'YYYY-MM-DDTHH:MM'."),
            "dropoff": ("datetime", True, None, "'YYYY-MM-DDTHH:MM'."),
            "car_class": ("enum:economy,compact,midsize,suv,minivan,luxury", False, "economy", "Car class."),
            "dropoff_location": ("text", False, None, "Return location; null means same as pickup."),
            "extras": ("set:gps,child_seat,ski_rack,additional_driver", False, [], "Add-ons."),
        },
    },
}


def _render_api() -> str:
    lines = []
    for name, spec in API.items():
        lines.append(f"{name}: {spec['description']}")
        for p, (typ, req, default, desc) in spec["params"].items():
            t = typ.replace("free", "string").replace("text", "string").replace("code", "string")
            if t.startswith("enum:"):
                t = "one of " + ", ".join(t[5:].split(","))
            elif t.startswith("set:"):
                t = "list, items from " + ", ".join(t[4:].split(","))
            elif t == "set":
                t = "list of strings"
            flag = "required" if req else f"optional, default {json.dumps(default)}"
            lines.append(f"  - {p} ({t}; {flag}): {desc}")
    return "\n".join(lines)


CONTEXT = (
    "You are a personal assistant that gets things done by calling the user's APIs.\n"
    f"Today is {TODAY}. The user is in the America/Los_Angeles time zone.\n"
    "The user's contacts:\n"
    + "\n".join(f"  - {n}: {e}" for n, e in CONTACTS.items())
    + "\n\nAvailable API functions:\n"
    + _render_api()
)

DELIVERABLE = (
    "Deliverable format: whenever you are ready to make the call, give exactly one call as "
    'JSON in a single ```json fenced block: {"name": "<function>", "arguments": {...}}. '
    "Leave out optional arguments to use their defaults."
)


TASKS = [
    Task(
        id="action_meeting",
        domain="action",
        title="Schedule a roadmap meeting",
        shards=[
            Shard(1, "set up a meeting with Priya about the Q2 roadmap"),
            Shard(2, "Make it Tuesday the 17th at 2pm."),
            Shard(3, "It should be 45 minutes long."),
            Shard(4, "Add Marco to it too."),
            Shard(5, "Put it in the Atlas conference room."),
            Shard(6, "Remind me 15 minutes before it starts."),
            Shard(7, "Can we push it to Wednesday the 18th, same time?", revises=2,
                  full_text="Make it Wednesday the 18th at 2pm."),
            Shard(8, "Add a video call link too, since some people are remote."),
            Shard(9, "Actually, swap Marco for Dana.", revises=4, full_text="Add Dana to it too."),
            Shard(10, "Priya is our head of product.", distractor=True),
        ],
        gold={"name": "create_event", "arguments": {
            "title": "Q2 roadmap", "start": "2026-03-18T14:00", "duration_minutes": 45,
            "attendees": ["priya.raman@acme.io", "dana.wu@acme.io"],
            "location": "Atlas", "reminder_minutes": 15, "video_call": True}},
        grading={"contains": {"location": "atlas"}},
        ablations={3: ("duration_minutes", 30), 5: ("location", None), 6: ("reminder_minutes", 10),
                   7: ("start", "2026-03-17T14:00"), 8: ("video_call", False),
                   9: ("attendees", ["priya.raman@acme.io", "marco.bianchi@acme.io"])},
    ),
    Task(
        id="action_flights",
        domain="action",
        title="Family flight search",
        shards=[
            Shard(1, "find me flights from SFO to Boston"),
            Shard(2, "Leaving April 2nd and coming back April 6th."),
            Shard(3, "It's for me and my two kids, so three passengers."),
            Shard(4, "Nonstop flights only."),
            Shard(5, "Premium economy."),
            Shard(6, "Sort the results by travel time rather than price."),
            Shard(7, "My partner's coming too, so make it 4 passengers.", revises=3,
                  full_text="It's for me, my partner and my two kids, so four passengers."),
            Shard(8, "Make the return April 7th instead.", revises=2,
                  full_text="Leaving April 2nd and coming back April 7th."),
            Shard(9, "It's for spring break.", distractor=True),
        ],
        gold={"name": "search_flights", "arguments": {
            "origin": "SFO", "destination": "BOS", "depart_date": "2026-04-02",
            "return_date": "2026-04-07", "passengers": 4, "cabin": "premium_economy",
            "max_stops": 0, "sort_by": "duration"}},
        ablations={4: ("max_stops", None), 5: ("cabin", "economy"), 6: ("sort_by", "price"),
                   7: ("passengers", 3), 8: ("return_date", "2026-04-06")},
    ),
    Task(
        id="action_restaurant",
        domain="action",
        title="Birthday dinner",
        shards=[
            Shard(1, "book a table at Nopa for dinner"),
            Shard(2, "This Saturday, March 14, at 7:30pm."),
            Shard(3, "There will be 4 of us."),
            Shard(4, "We'd like to sit outside."),
            Shard(5, "It's my partner's birthday."),
            Shard(6, "One person in the group is vegan."),
            Shard(7, "Actually my sister is coming too, so make it 5 people.", revises=3,
                  full_text="There will be 5 of us."),
            Shard(8, "It's supposed to be cold that night, so make it indoors instead.", revises=4,
                  full_text="We'd like to sit indoors."),
            Shard(9, "Someone also has a nut allergy."),
            Shard(10, "Nopa is the place on Divisadero.", distractor=True),
        ],
        gold={"name": "book_table", "arguments": {
            "restaurant": "Nopa", "date": "2026-03-14", "time": "19:30", "party_size": 5,
            "seating": "indoor", "occasion": "birthday", "dietary_restrictions": ["vegan", "nut_allergy"]}},
        grading={"contains": {"restaurant": "nopa"}},
        ablations={2: ("time", "19:00"), 5: ("occasion", "none"), 6: ("dietary_restrictions", ["nut_allergy"]),
                   7: ("party_size", 4), 8: ("seating", "outdoor"), 9: ("dietary_restrictions", ["vegan"])},
    ),
    Task(
        id="action_ticket",
        domain="action",
        title="File a checkout bug",
        shards=[
            Shard(1, "file a ticket about the checkout page crashing on Safari"),
            Shard(2, "It goes in the WEB project, and it's a bug."),
            Shard(3, "Priority is urgent; it's blocking purchases."),
            Shard(4, "Assign it to Dana."),
            Shard(5, "Tag it with the labels 'safari' and 'checkout'."),
            Shard(6, "It needs to be fixed by March 20."),
            Shard(7, "On second thought, make it high priority, not urgent. There's a workaround.", revises=3,
                  full_text="Priority is high (there's a workaround)."),
            Shard(8, "Add the label 'p1-customer' too."),
            Shard(9, "Assign it to Sam instead of Dana.", revises=4, full_text="Assign it to Sam."),
            Shard(10, "A customer reported this on social media.", distractor=True),
        ],
        gold={"name": "create_ticket", "arguments": {
            "project": "WEB", "title": "Checkout page crashes on Safari", "type": "bug",
            "priority": "high", "assignee": "sam.okafor@acme.io",
            "labels": ["safari", "checkout", "p1-customer"], "due_date": "2026-03-20"}},
        ablations={2: ("type", "task"), 5: ("labels", ["p1-customer"]), 6: ("due_date", None),
                   7: ("priority", "urgent"), 8: ("labels", ["safari", "checkout"]),
                   9: ("assignee", "dana.wu@acme.io")},
    ),
    Task(
        id="action_thermostat",
        domain="action",
        title="Bedroom heating schedule",
        shards=[
            Shard(1, "set up a thermostat schedule for the bedroom"),
            Shard(2, "Weekdays only."),
            Shard(3, "From 10:30pm until 6:00 in the morning."),
            Shard(4, "Keep it at 18 degrees."),
            Shard(5, "That's Celsius, not Fahrenheit."),
            Shard(6, "It should run in heating mode."),
            Shard(7, "Make it 19 degrees, actually.", revises=4, full_text="Keep it at 19 degrees."),
            Shard(8, "Include Saturday as well.", revises=2, full_text="Weekdays and Saturday."),
            Shard(9, "The bedroom is upstairs.", distractor=True),
        ],
        gold={"name": "set_thermostat_schedule", "arguments": {
            "zone": "bedroom", "days": ["mon", "tue", "wed", "thu", "fri", "sat"], "start_time": "22:30",
            "end_time": "06:00", "target_temp": 19, "unit": "C", "mode": "heat"}},
        ablations={3: ("start_time", "22:00"), 5: ("unit", "F"), 6: ("mode", "auto"),
                   7: ("target_temp", 18), 8: ("days", ["mon", "tue", "wed", "thu", "fri"])},
    ),
    Task(
        id="action_rental_car",
        domain="action",
        title="Ski-trip rental car",
        shards=[
            Shard(1, "I need a rental car when I land in Denver"),
            Shard(2, "Pick it up March 20 at 9am and drop it off March 23 at 5pm."),
            Shard(3, "An SUV; we're heading up to the mountains."),
            Shard(4, "We'll need a ski rack."),
            Shard(5, "And a child seat."),
            Shard(6, "I'll return it at the Boulder location instead of the airport."),
            Shard(7, "Change that to a minivan, more people are coming.", revises=3,
                  full_text="A minivan; more people are coming."),
            Shard(8, "Add a second driver."),
            Shard(9, "Never mind, I'll return it at the airport after all.", revises=6,
                  full_text="I'll return it at the airport where I picked it up."),
            Shard(10, "We're going skiing in Vail.", distractor=True),
        ],
        gold={"name": "rent_car", "arguments": {
            "pickup_location": "DEN", "pickup": "2026-03-20T09:00", "dropoff": "2026-03-23T17:00",
            "car_class": "minivan", "extras": ["ski_rack", "child_seat", "additional_driver"]}},
        grading={"aliases": {"pickup_location": ["den", "denver"]},
                 "aliases_null_ok": {"dropoff_location": ["den", "denver"]}},
        ablations={2: ("dropoff", "2026-03-23T09:00"), 4: ("extras", ["child_seat", "additional_driver"]),
                   5: ("extras", ["ski_rack", "additional_driver"]), 7: ("car_class", "suv"),
                   8: ("extras", ["ski_rack", "child_seat"]), 9: ("dropoff_location", "Boulder")},
    ),
]
