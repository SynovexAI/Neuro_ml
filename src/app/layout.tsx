import type { Metadata } from "next";
import { Analytics } from "@vercel/analytics/react";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Workbench",
  description: "Learn AI by building — a hands-on platform.",
};

// Apply the saved theme before paint so there's no flash. No stored choice →
// the CSS prefers-color-scheme media query drives it (system default).
const themeInit = `(function(){try{var t=localStorage.getItem('awb_theme');if(t==='light'||t==='dark'){document.documentElement.setAttribute('data-theme',t);}}catch(e){}})();`;

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head><script dangerouslySetInnerHTML={{ __html: themeInit }} /></head>
      <body>{children}<Analytics /></body>
    </html>
  );
}
