---
version: alpha
name: GenRAG
description: A general-purpose document learning assistant with a clean, modern interface for RAG-powered study and chat.
colors:
  primary: "#081A8A"
  secondary: "#1B35D6"
  tertiary: "#D8DAE3"
  neutral: "#FFFFFF"
  surface: "#FFFFFF"
  on-surface: "#000000"
  error: "#D93025"
  primary-60: "#2942C9"
  primary-70: "#10257F"
  primary-80: "#06145F"
typography:
  headline-display:
    fontFamily: "Times New Roman"
    fontSize: 32px
    fontWeight: 700
    lineHeight: 38px
    letterSpacing: 0px
  headline-lg:
    fontFamily: "Times New Roman"
    fontSize: 24px
    fontWeight: 700
    lineHeight: 29px
    letterSpacing: 0px
  headline-md:
    fontFamily: "Times New Roman"
    fontSize: 20px
    fontWeight: 600
    lineHeight: 24px
    letterSpacing: 0px
  headline-sm:
    fontFamily: "Times New Roman"
    fontSize: 18px
    fontWeight: 600
    lineHeight: 22px
    letterSpacing: 0px
  body-lg:
    fontFamily: "system-ui,ui-sans-serif,-apple-system,BlinkMacSystemFont,sans-serif,Inter,NotoSansHans"
    fontSize: 18px
    fontWeight: 400
    lineHeight: 30px
    letterSpacing: 0px
  body-md:
    fontFamily: "system-ui,ui-sans-serif,-apple-system,BlinkMacSystemFont,sans-serif,Inter,NotoSansHans"
    fontSize: 16px
    fontWeight: 400
    lineHeight: 27px
    letterSpacing: 0px
  body-sm:
    fontFamily: "system-ui,ui-sans-serif,-apple-system,BlinkMacSystemFont,sans-serif,Inter,NotoSansHans"
    fontSize: 14px
    fontWeight: 400
    lineHeight: 22px
    letterSpacing: 0px
  label-lg:
    fontFamily: "system-ui,ui-sans-serif,-apple-system,BlinkMacSystemFont,sans-serif,Inter,NotoSansHans"
    fontSize: 14px
    fontWeight: 400
    lineHeight: 20px
    letterSpacing: 0px
  label-md:
    fontFamily: "system-ui,ui-sans-serif,-apple-system,BlinkMacSystemFont,sans-serif,Inter,NotoSansHans"
    fontSize: 12px
    fontWeight: 500
    lineHeight: 16px
    letterSpacing: 0.01em
  label-sm:
    fontFamily: "Times New Roman"
    fontSize: 14px
    fontWeight: 400
    lineHeight: 18px
    letterSpacing: 0px
rounded:
  none: 0px
  sm: 4px
  md: 8px
  lg: 12px
  xl: 20px
  full: 9999px
spacing:
  xs: 8px
  sm: 16px
  md: 28px
  lg: 44px
  xl: 100px
components:
  button-primary:
    backgroundColor: "transparent"
    textColor: "{colors.on-surface}"
    typography: "{typography.label-lg}"
    rounded: "{rounded.full}"
    padding: "10px 12px"
    size: "108px"
    height: "40px"
  button-secondary:
    backgroundColor: "transparent"
    textColor: "{colors.on-surface}"
    typography: "{typography.label-lg}"
    rounded: "{rounded.full}"
    padding: "10px 12px"
    size: "108px"
    height: "40px"
  button-link:
    backgroundColor: "transparent"
    textColor: "{colors.on-surface}"
    typography: "{typography.label-sm}"
    rounded: "{rounded.none}"
    padding: "0px"
    size: "auto"
    height: "auto"
  card:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.on-surface}"
    typography: "{typography.body-md}"
    rounded: "{rounded.lg}"
    padding: "24px"
  input:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.on-surface}"
    typography: "{typography.body-md}"
    rounded: "{rounded.full}"
    padding: "10px 12px"
    height: "40px"
---

# GenRAG

## Overview

GenRAG is a general-purpose document learning assistant. Upload any PDF, ask questions grounded in your documents, and study with Chat, Learning, and Interview modes. The interface uses a deep navy palette with white content surfaces — polished, readable, and focused on learning rather than generic chat chrome.

## App layout

Two-panel layout:

- **Sidebar (left):** Documents, Memory, Mode selector, Debug pipeline status
- **Main (right):** Chat messages, source citations, input area

Background uses the primary navy gradient; chat and sidebar cards sit on white surfaces with tertiary borders.

## Colors

- **Primary (#081A8A):** Deep navy for app background and sidebar accent areas.
- **Secondary (#1B35D6):** Brighter blue for active states, links, and mode indicators.
- **Tertiary (#D8DAE3):** Borders and subtle separators on white cards.
- **Surface (#FFFFFF):** Chat area, cards, inputs.
- **On-surface (#000000):** Primary text.
- **Error (#D93025):** Errors and destructive actions.

## Typography

- **Headlines (Times New Roman):** Product name "GenRAG", section titles in sidebar.
- **Body (system sans):** Chat messages, labels, buttons.

## Components

- Pill-shaped buttons and inputs (`rounded.full`, 40px height).
- Cards: white, 1px tertiary border, `rounded.lg`, 24px padding.
- No heavy shadows — hierarchy via borders and color contrast.

## Do's and Don'ts

- Do use "GenRAG" as the product name everywhere in the UI.
- Do keep the sidebar informative (documents, memory, debug visibility).
- Do prefer borders over shadows.
- Don't use third-party brand names in the UI.
- Don't use dense cramped layouts — preserve spacing (8, 16, 28, 44px rhythm).
