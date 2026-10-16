# Separate context gathering from execution

Laban et al. (2025) showed that LLMs lose accuracy when a user gives the requirements of a task in pieces over many turns.
Most of that decrease came from unreliability, not from a loss of aptitude.
We test one fix for agents that must produce one definitive output, such as a Python function, a SQL query, an API call, or a number.
A gatherer agent talks with the user and writes the full spec again on each turn.
A stateless executor agent produces the deliverable from the spec only, so an early draft cannot anchor a later draft.
We compare this two-agent setup with one agent that has a plain prompt, and with a prompt-only control that tells one agent to collect the requirements first.

With current Claude models, the Laban protocol still decreases accuracy (Sonnet 5.5: 96.7% to 76.7%), but almost all of the decrease is on function-calling tasks.
On those tasks, most of the decrease comes from the form of the final answer, not from lost information.
On all 105 function-calling tasks from Laban et al., with Sonnet 5.5, the two-agent setup scored 90.4%.
The prompt-only control scored 80.0%, and the plain agent scored 51.2%.
The gain over the control is 10.4 points, with a 95% confidence interval of [+3.1, +18.5].
The two-agent setup costs about three times as much per correct answer: $0.052, against $0.017 for the control.

We also wrote 24 harder tasks in code, SQL, actions, and math, with 36 revisions in total.
With Haiku 5.5 and a more realistic simulated user, the accuracy of the plain agent decreased from 100% to 70.0%.
The prompt-only control scored 81.7%, the two-agent setup scored 84.2%, and a variant whose executor also sees the user's messages scored 88.3%.
On only 24 tasks, the difference between the architecture and the prompt is not significant (+2.5 points, CI [−5.8, +10.8]).
Thus the evidence is strongest where the final answer must contain the full deliverable.
For version 2, we propose to run the executor again each time that the spec changes.
