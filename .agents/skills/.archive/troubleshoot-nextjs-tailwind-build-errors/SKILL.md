---
name: troubleshoot-nextjs-tailwind-build-errors
description: Troubleshoot common build failures in Next.js projects that use Tailwind CSS, particularly issues with configuration files and PostCSS dependencies.
---

When a `npm run build` command fails for a Next.js application using Tailwind CSS, check for these common pitfalls before attempting more complex debugging.

### Pitfall 1: Incorrect `next.config.js` file extension

Modern JavaScript often uses the `.mjs` extension for ES modules. However, the Next.js build toolchain can be sensitive to this. If your build fails with a generic Webpack error, one of the first things to check is the `next.config` file.

**Symptom:**
The `npm run build` command fails with a generic `> Build failed because of webpack errors` message, often with a stack trace originating from `webpack/config` modules.

**Fix:**
Ensure your Next.js configuration file is named `next.config.js`, not `next.config.mjs`.

```bash
# If next.config.mjs exists, rename it
mv frontend/next.config.mjs frontend/next.config.js
```

### Pitfall 2: `autoprefixer` or `postcss` not found

Even if `autoprefixer` and `tailwindcss` are listed in `devDependencies` in `package.json`, the Next.js build process might fail to resolve them. This can happen if they are not installed as direct, explicit dependencies.

**Symptom:**
The build fails with `Error: Cannot find module 'autoprefixer'`.

**Fix:**
Install `postcss` and `autoprefixer` as explicit dependencies. This ensures they are available in `node_modules` when the build process looks for them.

```bash
cd frontend
npm install postcss autoprefixer
```

After installing, retry the build. If it still fails, consider a clean install to remove any stale state:

```bash
cd frontend
rm -rf .next node_modules
npm install
npm run build
```
