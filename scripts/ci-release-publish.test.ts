import { describe, expect, it } from "bun:test";
import { applyPublishBin } from "./ci-release-publish";

describe("release publish bin rewrite", () => {
	it("keeps the probe CLI as shipped TypeScript while omp uses the prepack bundle", async () => {
		const manifest = await applyPublishBin("packages/coding-agent", false);
		expect(manifest.bin).toEqual({
			omp: "dist/cli.js",
			"omp-openai-compat": "src/openai-compat-probe-cli.ts",
		});
	});
});
