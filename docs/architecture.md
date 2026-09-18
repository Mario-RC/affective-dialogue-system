# Architecture

The package separates model adapters, dialogue state, and response selection.
Imports do not download weights, allocate GPU memory, or start audio devices.
Constructing a neural service loads its model; construct services once and reuse
them across turns.

| Module | Responsibility |
| --- | --- |
| `config`, `factory` | Validate explicit configuration and construct services |
| `dialogue` | Build Gemma/Llama 3 prompts, generate continuations, validate emotional triples |
| `pipeline` | Classify emotion when needed and retain successful dialogue turns |
| `strategy` | Coordinate protocols, input filtering, templates, generator candidates, and fallback |
| `safety` | Rule-based checks and optional moderation adapters |
| `asr`, `tts` | Transcribe files and synthesize speech through optional libraries |
| `emotion`, `generation` | Emotion classification and auxiliary emotional GPT-2 |
| `interest` | Spanish keyword scoring with normalized whole-stem matching |

## State and validation

`AffectiveDialogueSystem.reply` appends a turn only after generation succeeds and
retains at most `max_history_turns`. Missing emotions use the injected classifier
or `NEUTRAL`. This API does not run response selection or moderation automatically.

The generation engine also bounds prompt history, requires completed historical
turns, and rejects prompts that exceed the model context window. It decodes only
the generated continuation. The parser requires exactly three recognized tags,
nonempty segments, the requested first/second emotions, and a final `NEUTRAL` tag.
Malformed output yields a localized fallback. The parser does not verify semantic
empathy, truthfulness, word count, or whether the final sentence is a question.

`DialogueContext` belongs to the application. The selector mutates protocol
metadata on that context; it does not append history, classify emotion, or infer
topics. Applications should store only accepted responses and maintain their own
per-conversation context. Generator callbacks receive deep copies so late or
rejected candidates cannot mutate the caller's state.

## Selection order

Handlers execute sequentially and stop at the first applicable response:

1. Glucose monitoring protocol.
2. Silence timeout.
3. Topic monitoring.
4. Input toxicity filtering.
5. Rule-based template response, checked for output toxicity.
6. Concurrent generator candidates, checked before delivery.
7. A predefined fallback.

Protocols return predefined messages and precede input moderation. Template
selection follows rule order; individual templates may randomly choose a reply.
Generic catch-all templates are skipped by default so generation remains reachable.
`RegexHandler` is available as a separate adapter, but is not part of the default
selection chain.

## Concurrency and lifecycle

Each selector owns one `LLMHandler`. It creates at most two worker threads and
permits only one outstanding job per source. Busy sources are skipped on later
turns, so repeated timeouts cannot build an unbounded queue. Results belong to the
request that created them; old results are never reused for a new message.

The first valid, safe candidate wins. Primary wins when both results are ready in
the same batch. Optional fallback translation and output checks run in the worker
and count toward the wait deadline. Generator, translator, or detector failures
reject that candidate and are logged without message contents.

Use `with SelectingStrategy(...) as strategy` or call `strategy.close()` to stop
accepting jobs. `close(wait_for_running=True)` waits for active jobs to finish.
A Python thread cannot forcibly stop running inference: a timed-out job may still
consume resources, and Python waits for executor threads when exiting. Hard
cancellation requires an external inference service or process isolation. This
matches the [Python executor lifecycle](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Executor.shutdown).

Callbacks should consume ordinary copyable context data and return a string,
`AssistantResponse`, `StrategyResult`, or `None`. State changes must be applied by
the caller after selection, not inside a generator. Share model instances across
concurrent callbacks only when the adapter supports that access pattern.

## Moderation

The default detector uses packaged terms, phrases, and category patterns. It is a
heuristic, not a guarantee of safety. Title allowlisting uses phrase boundaries and
does not suppress unrelated matches elsewhere in the message.

The filter checks user input and generated text separately. Adapters that implement
`check_response` receive assistant output through that method. Llama Guard 2 uses
its documented S1–S11 taxonomy; unknown or malformed model verdicts are flagged.
See the [model's taxonomy](https://huggingface.co/meta-llama/Meta-Llama-Guard-2-8B#harm-taxonomy-and-policy).

## Research glucose protocol

The state machine preserves the prototype's interaction policy and thresholds;
this repository does not establish their clinical validity. Tests cover software
state transitions and data handling, not treatment effectiveness or medical safety.

Initial `glucose_level` is supplied by the application in mg/dL. Measurement
responses must contain a positive number, optionally followed by `mg/dL`. The
handler does not extract arbitrary numbers from prose or reuse the initial reading
as a new measurement. A received measurement updates the context, including on
completion, so a recovered reading does not immediately retrigger the protocol.
The application must handle `should_end_session` and any real-world escalation.
