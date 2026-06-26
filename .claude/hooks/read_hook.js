const path = require("path");

process.stdin.setEncoding("utf8");
let input = "";
process.stdin.on("data", (d) => (input += d));
process.stdin.on("end", () => {
  const toolArgs = JSON.parse(input);
  const rawPath = toolArgs.tool_input?.file_path || "";

  // Resolve to an absolute, normalized path before checking
  const resolved = path.resolve(rawPath);
  const base = path.basename(resolved);

  // Match .env, .env.local, .env.production, etc., but not unrelated files
  const isEnvFile = /^\.env(\..+)?$/i.test(base);

  if (isEnvFile) {
    console.error(`Blocked: cannot read env file (${resolved})`);
    process.exit(2);
  }
  process.exit(0);
});