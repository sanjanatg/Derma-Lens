```markdown
# Design System Document: The Tactile Apothecary

## 1. Overview & Creative North Star
This design system is built upon the "Tactile Apothecary" North Star. In the high-stakes world of skin health, we must reject the cold, sterile aesthetics of traditional medical apps. Instead, we embrace a high-end editorial approach that feels like a premium skincare journal: warm, human, and meticulously intentional.

By utilizing "Organic Minimalism," we break the standard mobile template. We prioritize breathing room over information density and use asymmetrical layouts to guide the eye naturally. The goal is to transform a moment of potential anxiety (scanning for skin cancer) into a moment of calm, professional care.

---

## 2. Colors: Tonal Depth & Soul
Our palette is rooted in the earth. It is designed to feel "grown," not "manufactured."

*   **Primary (`primary` #5a6241 / `primary_container` #a7b08a):** Muted Sage is our color of growth and health. Use the container variant for large action areas to maintain a soft, approachable feel.
*   **Surface (`surface` #fcf9f2 / `surface_container` #f1eee7):** Our "Linen White" base. It provides a warm, paper-like texture to the digital canvas.
*   **On-Surface (`on_surface` #1c1c18):** A deep, organic brown—never pure black. This maintains high legibility while feeling softer on the eyes.

### The "No-Line" Rule
**Strict Mandate:** Designers are prohibited from using 1px solid borders to define sections. We define boundaries through tonal shifts.
*   **The Transition:** If a list needs to be separated from the background, place it on a `surface_container_low` block. Use the change in background color to tell the user where one thought ends and another begins.

### Surface Hierarchy & Nesting
Think of the UI as a series of stacked, premium cardstock. 
1.  **Level 0 (Base):** `surface`
2.  **Level 1 (Sectioning):** `surface_container`
3.  **Level 2 (Interaction):** `surface_container_high` (used for active states or cards)

### Signature Textures & Glassmorphism
To elevate the experience beyond "flat," use semi-transparent overlays for floating navigation or modals. 
*   **Glass Specs:** `surface` color at 70% opacity with a 16px backdrop-blur. 
*   **Gradients:** Use subtle linear gradients from `primary` to `primary_container` (at a 135° angle) for hero-level CTAs to add a "soulful" glow.

---

## 3. Typography: Editorial Authority
We use **Plus Jakarta Sans** for its friendly, rounded geometry. It strikes a balance between professional medical trust and approachable warmth.

*   **Display Scale (`display-lg` 3.5rem):** Reserved for emotional "Hero" moments (e.g., "Your skin is looking clear"). Use with wide letter-spacing (-2%).
*   **Headline Scale (`headline-md` 1.75rem):** For page titles. These should often be left-aligned with significant top-margin (`spacing.16`) to create an editorial feel.
*   **Body Scale (`body-lg` 1rem):** Used for all medical results and instructions. Line height must be generous (1.5x) to ensure readability for all age groups.
*   **Labels (`label-md` 0.75rem):** Use `on_surface_variant` (#46483e) to distinguish metadata from actionable text.

---

## 4. Elevation & Depth: Tonal Layering
Traditional "drop shadows" are too aggressive for this system. We use light to imply depth.

*   **The Layering Principle:** Depth is achieved by "stacking." A `surface_container_lowest` card placed on a `surface_container` background creates a soft, natural lift without a single shadow.
*   **Ambient Shadows:** For elements that truly "float" (like a camera trigger), use an ultra-diffused shadow:
    *   **Color:** `on_surface` at 6% opacity.
    *   **Blur:** 24px.
    *   **Y-Offset:** 8px.
*   **The Ghost Border:** If accessibility requires a stroke (e.g., in high-contrast modes), use `outline_variant` (#c7c7bb) at **15% opacity**. It should be a whisper, not a shout.

---

## 5. Components: Bespoke Elements

### Buttons (Pill-Shaped)
All buttons use `rounded.full`. 
*   **Primary:** Background `primary_container` (#A7B08A), Text `on_primary_container` (#3B4325).
*   **Secondary:** Background `surface_container_highest`, Text `on_surface`.
*   **States:** On press, scale the button down slightly (98%) rather than just changing color. This adds a tactile, physical feel.

### Input Fields (The "Soft Well")
Forbid the standard boxed input. 
*   **Style:** Use a `surface_container_low` background with a `rounded.md` corner. 
*   **Interaction:** On focus, the background shifts to `surface_container_highest` and the label moves up using `label-sm`. No high-contrast borders.

### Cards & Lists: The "Breath" Rule
*   **Forbid Dividers:** Do not use horizontal lines between list items.
*   **Separation:** Use `spacing.4` or `spacing.6` of vertical white space. If items must be grouped, wrap them in a single `surface_container` shape.

### Specialty Component: The Scanning Ring
For the skin-scanning interface, use a "Frosted Ring." A semi-transparent `primary_fixed_dim` circle with a `backdrop-blur` of 8px. This guides the user's focus while keeping the skin visible and integrated into the UI.

---

## 6. Do's and Don'ts

### Do
*   **Do** use asymmetrical margins (e.g., `spacing.8` on the left, `spacing.12` on the right) for header text to create an editorial look.
*   **Do** prioritize the `surface` color. 80% of the screen should feel like "Linen White."
*   **Do** use `rounded.xl` or `rounded.full` for all containers to reinforce the "soft/friendly" brand pillar.

### Don't
*   **Don't** use 100% black text. Always use the organic brown `on_surface` (#1c1c18).
*   **Don't** use standard Material Design "Elevated Cards" with heavy shadows. Stick to tonal layering.
*   **Don't** cram information. If a screen feels busy, increase the spacing scale (e.g., move from `spacing.4` to `spacing.8`).
*   **Don't** use sharp corners. Nothing in the "Tactile Apothecary" should feel like it could prick or scratch.

---
**Director’s Closing Note:** 
Remember, we are designing for someone who might be worried. Every pixel must work to lower their heart rate. Use the "Linen White" to give them space to breathe, and the "Muted Sage" to give them the confidence of professional care.```