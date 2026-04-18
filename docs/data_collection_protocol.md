# Data Collection Protocol

## Scope

This protocol defines how new gesture recordings should be captured for the unified LSTM pipeline.

## Recording Format

- Keep the per-timestep sensor CSV numeric only.
- Store all capture metadata in a sidecar manifest file with the same sample stem.
- Do not append orientation, posture, or other string metadata into the sensor matrix.

## Static Gesture Capture Rules

- Record each static gesture at multiple physical orientations.
- Keep the internal capture window still so the model learns a stable pose signature.
- Vary macro-position across captures, including arm height and wrist orientation, while preserving intra-window stillness.
- Record balanced coverage across the orientation set before finalizing a class.

## REST Training Rules

- Include true resting state samples.
- Include dynamic return-to-baseline sequences after gesture completion.
- Keep REST as a learned class, not only as a runtime fallback.

## Metadata Handling

- Use sidecar metadata for orientation, posture, sample ID, and session context.
- Preserve the numeric CSV schema so downstream sequence serialization remains compatible with NumPy float32 arrays.

## Expected Outcome

- Static gestures retain spatial diversity without contaminating the sensor tensor schema.
- REST examples cover both idle and return-to-baseline behavior.
- Old recordings remain usable as legacy numeric-only captures.
