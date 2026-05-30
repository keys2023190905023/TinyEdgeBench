# TinyEdgeBench Frontend Design Rules

These rules encode the UI quality bar for TinyEdgeBench. They are based on the installed Figma/design-system workflow skills and apply to the static GitHub Pages site in `docs/` and the local Streamlit UI in `src/tinyedgebench/web_app.py`.

## Product Direction

- Aim for a first-tier technology product feel: calm, precise, premium, and highly readable.
- Prefer Apple-like restraint over decorative cyberpunk styling: generous whitespace, soft material layers, subtle motion, crisp typography, and restrained blue system accents.
- The benchmark app is an engineering tool, so density should support scanning and repeated use. Avoid marketing clutter inside the tool surface.
- The static site may feel more editorial, but the product itself must remain the first-viewport signal.

## Design Tokens

- IMPORTANT: Use CSS variables for colors, surfaces, borders, shadows, and semantic accents. Do not scatter new hardcoded colors through components.
- Use a stable type scale. Do not scale font size directly with viewport width. Use media queries for responsive type changes.
- Letter spacing must remain `0` unless there is a very specific short label that benefits from uppercase spacing.
- Keep radii consistent: use 999px for pills, 16-28px for premium material surfaces, and smaller radii only for dense control internals.
- Use subtle shadows and translucency. Avoid heavy neon glow, decorative blobs, or noisy background effects.

## Layout And Components

- Build sections from clear primitives: hero, product metric strip, backend matrix, proof row, catalog, workflow, install terminal, and report/results surfaces.
- Do not nest cards inside cards. Use section bands and unframed layout for major page structure.
- Ensure all fixed-format UI elements have stable dimensions so hover states and dynamic text do not shift layout.
- Buttons should be command-like and obvious: primary actions use the blue system accent, secondary actions use neutral glass or outline styling.
- Backend status cards must distinguish ready and pending states with text and color, not color alone.

## Accessibility And QA

- Maintain sufficient contrast in both the dark hero and light product sections.
- Every image must have meaningful alt text.
- Validate desktop and mobile layouts after frontend changes.
- Check the browser console for errors after every significant UI change.
- Prefer Browser or Playwright-based screenshots/snapshots for visual QA. Do not rely only on code inspection.

## Figma-To-Code Workflow

- If a Figma design is provided later, fetch design context and screenshot first, then implement section by section.
- Reuse existing project tokens and layout conventions rather than copying arbitrary generated styles.
- Validate against the Figma screenshot for spacing, typography, color, and responsive behavior before shipping.
