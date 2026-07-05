# Agent Task 06 — Six-channel lamp simulator

Child model: `ollama/qwen3-coder:30b` launched as patch-proposal process `proc_100073b43b83`.

## RED

Command:

```bash
pnpm --dir frontend exec vitest --run src/scene/pose.test.ts
```

Expected failure observed:

```text
Cannot find module './pose'
```

## GREEN

Commands:

```bash
pnpm --dir frontend test -- --run src/scene/pose.test.ts
pnpm --dir frontend exec tsc --noEmit
pnpm --dir frontend build
```

Results:

- Vitest: 1 passed
- TypeScript: passed after adding `@types/three`
- Vite build: passed with standard large chunk warning from Three/R3F bundle
