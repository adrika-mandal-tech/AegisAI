import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "AegisAI AgentSwarm — Autonomous Multi-Agent Intelligence Workspace",
  description:
    "Eight-agent autonomous AI research, document intelligence, dataset analytics, visualization, and deliverable generation platform.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="workspace-bg" aria-hidden="true" />
        {children}
      </body>
    </html>
  );
}
