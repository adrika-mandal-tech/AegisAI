"use client";

import { Github, Linkedin, Mail } from "lucide-react";

export type DeveloperLinkKey = "github" | "linkedin" | "gmail";

export const DEVELOPER_URLS = {
  github: "https://github.com/adrika-mandal-tech/AegisAI",
  linkedin: "https://www.linkedin.com/in/adrika-mandal-753226246/",
  gmail: "mailto:adrika.mandal01@gmail.com",
} as const;

interface NavbarSocialIconsProps {
  activeKey?: DeveloperLinkKey | null;
  onSelect?: (key: DeveloperLinkKey) => void;
  size?: "sm" | "md";
  className?: string;
}

/**
 * Circular navbar icon links for GitHub, LinkedIn, and Gmail.
 * Clicking any icon highlights it with the AegisAI Lemon yellow accent.
 */
export function NavbarSocialIcons({
  activeKey,
  onSelect,
  size = "md",
  className = "",
}: NavbarSocialIconsProps) {
  const circleDim = size === "sm" ? "w-7 h-7" : "w-8 h-8";
  const githubIconDim = size === "sm" ? "w-3.5 h-3.5" : "w-4 h-4";
  const compactIconDim = size === "sm" ? "w-3 h-3" : "w-3.5 h-3.5";

  const buttonClass = (key: DeveloperLinkKey) => {
    const isClicked = activeKey === key;
    return [
      circleDim,
      "rounded-full inline-flex items-center justify-center border shrink-0 transition-all duration-150",
      "active:bg-[#F3E66B] active:border-[#D4BF2A] active:text-[#272436]",
      isClicked
        ? "bg-[#F3E66B] border-[#D4BF2A] text-[#272436] shadow-sm"
        : "bg-[#FAFAF4] border-[#EAE7EF] text-[#4F3685] hover:bg-[#F0EAFB] hover:border-[#D8CCF2] hover:text-[#272436] hover:-translate-y-0.5",
    ].join(" ");
  };

  return (
    <div className={`flex items-center gap-1.5 ${className}`}>
      <a
        href={DEVELOPER_URLS.github}
        target="_blank"
        rel="noopener noreferrer"
        title="View NexusAI on GitHub"
        aria-label="View NexusAI on GitHub"
        onClick={() => onSelect?.("github")}
        className={buttonClass("github")}
      >
        <Github className={githubIconDim} />
      </a>

      <a
        href={DEVELOPER_URLS.linkedin}
        target="_blank"
        rel="noopener noreferrer"
        title="LinkedIn"
        aria-label="LinkedIn"
        onClick={() => onSelect?.("linkedin")}
        className={buttonClass("linkedin")}
      >
        <Linkedin className={compactIconDim} />
      </a>

      <a
        href={DEVELOPER_URLS.gmail}
        title="Gmail"
        aria-label="Gmail"
        onClick={() => onSelect?.("gmail")}
        className={buttonClass("gmail")}
      >
        <Mail className={compactIconDim} />
      </a>
    </div>
  );
}

interface ContactDeveloperProps {
  activeKey?: DeveloperLinkKey | null;
  onSelect?: (key: DeveloperLinkKey) => void;
  variant?: "sidebar" | "card";
}

/**
 * Contact Developer action group featuring LinkedIn and Gmail with circle logos.
 * Displays "Gmail" rather than the raw address and turns yellow when clicked.
 */
export function ContactDeveloper({
  activeKey,
  onSelect,
  variant = "sidebar",
}: ContactDeveloperProps) {
  const pillClass = (key: DeveloperLinkKey) => {
    const isClicked = activeKey === key;
    return [
      "inline-flex items-center gap-2 rounded-xl border font-semibold transition-all duration-150",
      variant === "sidebar" ? "flex-1 justify-center px-2.5 py-1.5 text-[11px]" : "px-4 py-2.5 text-xs",
      "active:bg-[#F3E66B] active:border-[#D4BF2A] active:text-[#272436]",
      isClicked
        ? "bg-[#F3E66B] border-[#D4BF2A] text-[#272436] shadow-sm"
        : "bg-[#FAFAF4] border-[#EAE7EF] text-[#272436] hover:bg-[#F0EAFB] hover:border-[#D8CCF2] hover:text-[#4F3685]",
    ].join(" ");
  };

  const iconWrapClass = (key: DeveloperLinkKey) => {
    const isClicked = activeKey === key;
    return [
      variant === "sidebar" ? "w-5 h-5" : "w-6 h-6",
      "rounded-full inline-flex items-center justify-center shrink-0 transition-colors",
      isClicked ? "bg-[#FFF9CC] text-[#272436]" : "bg-[#F0EAFB] text-[#4F3685]",
    ].join(" ");
  };

  return (
    <div className={variant === "sidebar" ? "space-y-2" : "flex flex-wrap gap-3"}>
      {variant === "sidebar" && (
        <div className="text-[11px] font-bold uppercase tracking-wider text-[#777383]">
          Contact Developer
        </div>
      )}
      <div className={variant === "sidebar" ? "flex items-center gap-2" : "flex flex-wrap items-center gap-3"}>
        <a
          href={DEVELOPER_URLS.linkedin}
          target="_blank"
          rel="noopener noreferrer"
          title="LinkedIn"
          aria-label="LinkedIn"
          onClick={() => onSelect?.("linkedin")}
          className={pillClass("linkedin")}
        >
          <span className={iconWrapClass("linkedin")}>
            <Linkedin className={variant === "sidebar" ? "w-3 h-3" : "w-3.5 h-3.5"} />
          </span>
          <span>LinkedIn</span>
        </a>

        <a
          href={DEVELOPER_URLS.gmail}
          title="Gmail"
          aria-label="Gmail"
          onClick={() => onSelect?.("gmail")}
          className={pillClass("gmail")}
        >
          <span className={iconWrapClass("gmail")}>
            <Mail className={variant === "sidebar" ? "w-3 h-3" : "w-3.5 h-3.5"} />
          </span>
          <span>Gmail</span>
        </a>
      </div>
    </div>
  );
}
