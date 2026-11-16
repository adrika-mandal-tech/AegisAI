---
name: nextjs-configure-force-casing
description: When setting up a Next.js project, enable the `forceConsistentCasingInFileNames` option in `tsconfig.json` for better cross-platform compatibility.
---

# Configure `forceConsistentCasingInFileNames` in Next.js

When creating a `tsconfig.json` file for a new Next.js project, it is a best practice to enable the `forceConsistentCasingInFileNames` compiler option. This setting helps prevent file-casing issues that can occur when developing across different operating systems (like macOS/Windows and Linux), which handle case-sensitivity in file paths differently.

## Procedure

When creating or modifying a `tsconfig.json` file, add this line to the `compilerOptions` section:

```json
{
  "compilerOptions": {
    // ... other options
    "forceConsistentCasingInFileNames": true,
    // ... other options
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
```

This is often flagged as a warning by the IDE, and it's good practice to address it proactively.

### Example

During a session, the IDE raised a warning after `tsconfig.json` was created.

**IDE Warning:** `The compiler option \"forceConsistentCasingInFileNames\" should be enabled to reduce issues when working with different OSes.`

The agent then correctly added the option to the file.
