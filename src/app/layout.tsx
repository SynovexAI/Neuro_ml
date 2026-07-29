import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Workbench",
  description: "Learn AI by building — a hands-on platform.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
