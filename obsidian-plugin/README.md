# Lorekeeper — Obsidian plugin (M3 stub)

A thin panel that talks to the local `lorekeeper` MCP server.

## Status
Stub: commands load and the server URL is configurable, but they don't call the server
yet. M3 wires `suggest_placement` and a chat view to `python -m lorekeeper.server --http`.

## Dev
```powershell
cd D:\dev\lorekeeper\obsidian-plugin
npm install
npm run dev        # watches main.ts -> main.js
```
Then symlink/copy this folder into `<vault>\.obsidian\plugins\lorekeeper\` and enable it
in Obsidian (Settings -> Community plugins).
