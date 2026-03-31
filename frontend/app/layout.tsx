import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Speech Separation Lab",
  description: "Input pipeline for multi-speaker speech separation",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
