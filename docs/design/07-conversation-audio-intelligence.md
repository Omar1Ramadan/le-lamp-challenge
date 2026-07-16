# Conversation and Audio Intelligence

## Responsibility

Provide voice and text interaction, read-only grounded memory tools, interruption handling, anonymous active-speaker association, coarse vocal-affect evidence, and an offline fallback. Conversation does not own memory or behavior state.

## Provider Port

```text
ConversationProvider
  async start(session_context) -> ProviderSession
  async handle_text(turn_id, text, tools) -> streamed ResponseChunk
  async handle_audio(turn_id, audio_stream, tools) -> streamed ResponseChunk
  async interrupt(turn_id, reason) -> InterruptResult
  async close(reason)
```

The initial cloud adapter targets the configured OpenAI realtime-capable API. Model name, endpoint, and voice are explicit configuration values rather than source constants. `TemplateConversationProvider` is always available and uses deterministic parsing plus answer templates for text input and replay.

Provider output is untrusted until it passes grounding validation. The application records provider timing and tool calls but never secrets or raw authentication headers.

## Conversation Turn Flow

1. Voice activity or submitted text creates a turn and correlation ID.
2. Speech-to-text produces an incremental transcript when voice is used.
3. The provider may call one of the allowed memory tools.
4. The application executes the typed query and returns bounded evidence.
5. The provider phrases an answer.
6. A grounding validator confirms every factual object, time, and location field exists in tool evidence.
7. Valid text is displayed and optionally synthesized. Invalid output is replaced by a deterministic evidence template.
8. Policy receives `think`, `recall_success`, or `recall_unknown` events; it never receives free-form motor instructions.

## Allowed Tools

- `find_last_seen(object_label, session_scope, before_utc?)`
- `find_location(object_label, session_scope)`
- `list_recent_observations(object_label?, limit<=10)`

Tools return the `MemoryResult` contract. There is no arbitrary SQL, filesystem, network, memory-write, behavior, or adapter tool. Query inputs are length-limited and normalized before execution.

## Grounding Rules

- A factual answer must retain selected observation IDs in turn metadata.
- Object label, time, horizontal region, depth band, and anchor may appear only when supplied by evidence.
- Relative phrases such as “earlier” are resolved from evidence timestamps, not model assumptions.
- `ambiguous` results cause a clarification question listing bounded alternatives.
- `not_found`, low-confidence, or stale results produce an explicit uncertainty response.
- The UI can reveal the supporting memory card for any answer.

## Voice Activity and Interruption

Audio is processed in 20 ms chunks. Voice activity starts after 120 ms of speech probability above threshold and ends after 500 ms of silence. Thresholds are configurable and captured in replay manifests.

Speech detected while lamp audio is playing calls `interrupt` after at least 250 ms of direct speech above confidence threshold, cancels adapter audio, sets `audio_mode=listening`, and applies a 1,000 ms interruption cooldown to prevent cancellation spam. Partial transcripts and interruption timestamps remain in the trace.

## Background-Media Suppression

An audio classifier estimates `direct_speech`, `conversation_background`, `television_media`, `music`, or `other`. Visual speaker association and current engagement modify confidence. Confident television/media blocks unsolicited lamp sound, updates VAD status as `background_media`, and does not activate listening. It does not prevent explicit typed queries.

False certainty is avoided: when source classification is weak, the world model reports `unknown` and the policy chooses the quieter behavior.

## Multi-User Speaker Association

The subsystem prefers explicit classifier `speaker_id` values when available. Without directional audio, one visible person is marked as active speaker; with multiple visible people, the current primary person may be used as a heuristic. Named identity is never inferred or persisted. Conversation context may use Person A/B labels only within the session.

## Coarse Vocal Affect Bonus

After at least three seconds and at most eight seconds of voiced audio, local prosodic features or a configured classifier emit:

```text
VocalAffectObservation
  valence_tendency: float [-1, 1]
  arousal: float [0, 1]
  confidence: float [0, 1]
  window_ms, speaker_id?, feature_version
```

Policy ignores confidence below `0.60`. Affect is displayed as “vocal signal tendency,” is not retained beyond the session, and does not influence factual recall.

## Offline and Failure Behavior

- Missing cloud configuration selects `TemplateConversationProvider` at startup.
- Network loss during a turn interrupts cloud output and offers the deterministic text fallback using any completed memory result.
- Speech recognition unavailable leaves typed input active.
- Text-to-speech unavailable displays text and uses light/motion only.
- Tool timeout returns a grounded unavailable response; the provider cannot retry more than once per turn.
- Provider output failing grounding validation is replaced, recorded as a validation fault, and never spoken.
- Background classifier or affect failure disables that bonus only.

## Privacy

Raw microphone data remains local unless required by the configured cloud speech provider. The dashboard clearly indicates cloud audio use. Default traces retain transcripts and timing but not raw audio; evaluation fixtures require explicit recording mode. Session end deletes affect state and speaker associations.

## Tests

- Tool allowlist, argument bounds, alias normalization, and timeout behavior.
- Grounding validator acceptance and rejection for every evidence field.
- Not-found, ambiguous, stale, and low-confidence answer templates.
- Voice start/end timing and interruption while audio is buffered.
- Background-media suppression and uncertain-source fallback.
- Anonymous speaker association and session cleanup.
- Affect confidence gate and policy-safe ranges.
- Complete offline operation with cloud networking disabled.
