import esbuild from "esbuild";

const prod = process.argv.includes("production");

// Obsidian provides these at runtime; Node builtins must not be bundled.
const external = ["obsidian", "electron", "child_process", "fs", "path", "os", "util", "events", "stream"];

const ctx = await esbuild.context({
  entryPoints: ["main.ts"],
  bundle: true,
  external,
  format: "cjs",
  platform: "node",
  target: "es2018",
  sourcemap: prod ? false : "inline",
  outfile: "main.js",
  logLevel: "info",
});

if (prod) {
  await ctx.rebuild();
  await ctx.dispose();
  process.exit(0);
} else {
  await ctx.watch();
}
