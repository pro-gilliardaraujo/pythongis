---
name: Global Development Standards
description: Comprehensive guidelines for code quality, architecture, and maintainability across all projects.
---
# Global Development Standards & Architecture

This skill serves as the "Master Rulebook" for all projects managed by the Setup. It combines Senior Mentorship, Clean Code, and Architectural Strategy.

## 1. The Human-Centric Code Principle
*   **Code is for Humans**: We write code to be read by humans first, machines second.
*   **No "AI-Only" Logic**: Avoid compressed, cryptic, or generated-looking variable names (e.g., `x1`, `proc_gen_2`).
*   **Self-Documenting**: Functions should explain *what* they do via their name.
    *   *Bad*: `handleData()`
    *   *Good*: `processUserRegistrationData()`

## 2. Architecture: Composition Over Repetition
*   **The DRY Rule (Don't Repeat Yourself)**: If you copy-paste a file and change 10%, you have failed.
*   **Pattern**: Use **Generic Components** + **Specific Configurations**.
    *   *Scenario*: You need 5 different tables (Users, Products, Orders).
    *   *Wrong*: Create `UserTable.tsx`, `ProductTable.tsx`, `OrderTable.tsx` (all with copy-pasted sorting/filtering logic).
    *   *Right*: Create `DataTable.tsx` (handles sorting/filtering). Create `userColumns`, `productColumns`. Render `<DataTable data={users} columns={userColumns} />`.

## 3. Component Design
*   **Single Responsibility**: A component should do one thing.
    *   *Anti-Pattern*: A "Modal" file that contains the Dialg UI, the Form Inputs, the Validation Schema, and the API Submit logic.
    *   *Solution*: Split into `UserForm.tsx` (Inputs), `UserSchema.ts` (Validation), and `UserModal.tsx` (Wrapper).

## 4. Maintenance & Scalability
*   **Centralize Logic**: If logic is pivotal (like "How we filter dates"), it belongs in a utility function (`dateUtils.ts` or `useDateFilter`), not inside a UI component.
*   **Comment the "Why"**: Don't comment `// Sets state to true`. Comment `// Open modal to allow immediate editing after creation`.
