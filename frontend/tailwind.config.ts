import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        ivory: "#FAFAF4",
        surface: "#FFFFFF",
        lemon: {
          DEFAULT: "#F3E66B",
          soft: "#FFF9CC",
          border: "#E6D54F",
          ink: "#5C4900",
        },
        plum: {
          DEFAULT: "#7454B8",
          deep: "#4F3685",
          soft: "#F0EAFB",
          border: "#D8CCF2",
        },
        ink: {
          DEFAULT: "#272436",
          muted: "#777383",
          line: "#EAE7EF",
        },
        aegis: {
          50: "#F0EAFB",
          100: "#E4D9F7",
          200: "#CBB6EF",
          300: "#AE8DE3",
          400: "#916FD0",
          500: "#7454B8",
          600: "#5F429E",
          700: "#4F3685",
          800: "#3B2864",
          900: "#272436",
        },
      },
      boxShadow: {
        soft: "0 4px 20px -2px rgba(39, 36, 54, 0.05), 0 1px 3px rgba(39, 36, 54, 0.04)",
        card: "0 8px 28px -6px rgba(79, 54, 133, 0.08), 0 2px 6px rgba(39, 36, 54, 0.04)",
      },
    },
  },
  plugins: [],
};
export default config;
