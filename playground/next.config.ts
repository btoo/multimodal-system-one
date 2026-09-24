import type { NextConfig } from "next";
import { withWorkflow } from "workflow/next";
// The SDK's implicit local default uses .next/workflow-data, which a build
// replaces. Keep local run records outside build output, as on Vercel.
if (!process.env.VERCEL_DEPLOYMENT_ID) {
  process.env.WORKFLOW_TARGET_WORLD ??= "local";
  process.env.WORKFLOW_LOCAL_DATA_DIR ??= ".workflow-data";
}
const config: NextConfig = {
  agentRules: false,
  poweredByHeader: false,
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
          { key: "X-Frame-Options", value: "DENY" },
        ],
      },
    ];
  },
};
export default withWorkflow(config);
