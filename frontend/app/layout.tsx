import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { Nav } from "@/components/nav";
import "./globals.css";

const inter = Inter({ variable: "--font-inter", subsets: ["latin"] });
const mono = JetBrains_Mono({ variable: "--font-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "IPOPilot AI",
  description: "Deep-learning IPO listing forecasts, allotment probabilities and capital-allocation plans for Indian IPOs.",
};

const themeBoot = `try{var t=localStorage.getItem('ipopilot-theme');if(t)document.documentElement.setAttribute('data-theme',t)}catch(e){}`;

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head><script dangerouslySetInnerHTML={{ __html: themeBoot }} /></head>
      <body className={`${inter.variable} ${mono.variable} antialiased`}>
        <Nav />
        <main className="mx-auto w-full max-w-7xl px-4 pb-16 pt-6 md:px-6">{children}</main>
        <footer className="mx-auto max-w-7xl px-4 pb-10 text-xs leading-relaxed text-ink-3 md:px-6">
          IPOPilot AI is an educational research and decision-support tool — not investment advice and not a SEBI-registered
          investment adviser. GMP is unofficial; forecasts are probabilistic and can be wrong. Official data © NSE · BSE · SEBI.
        </footer>
      </body>
    </html>
  );
}
