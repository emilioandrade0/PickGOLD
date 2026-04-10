import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { getPublicAppSettings } from "../services/auth.js";

const AppSettingsContext = createContext({
  socialMode: false,
  setSocialMode: () => {},
  uiTheme: "original",
  setUiTheme: () => {},
  refreshSettings: async () => ({ ok: false }),
});

const UI_THEME_STORAGE_KEY = "pickgold_ui_theme";

function normalizeUiTheme(value) {
  const txt = String(value || "").trim().toLowerCase();
  if (txt === "dashboard_pro") return "dashboard_pro";
  return "original";
}

function getStoredUiTheme() {
  if (typeof window === "undefined") return "original";
  return normalizeUiTheme(window.localStorage.getItem(UI_THEME_STORAGE_KEY));
}

export function AppSettingsProvider({ children }) {
  const [socialMode, setSocialMode] = useState(false);
  const [uiThemeState, setUiThemeState] = useState(() => getStoredUiTheme());

  const setUiTheme = useCallback((value) => {
    const next = normalizeUiTheme(value);
    setUiThemeState(next);
    if (typeof window !== "undefined") {
      window.localStorage.setItem(UI_THEME_STORAGE_KEY, next);
    }
  }, []);

  const refreshSettings = useCallback(async () => {
    const res = await getPublicAppSettings();
    if (res?.ok) {
      setSocialMode(Boolean(res.social_mode));
      if (res.ui_theme) {
        setUiTheme(res.ui_theme);
      }
    }
    return res;
  }, [setUiTheme]);

  useEffect(() => {
    let active = true;

    async function loadSettings() {
      const res = await getPublicAppSettings();
      if (active && res?.ok) {
        setSocialMode(Boolean(res.social_mode));
        if (res.ui_theme) {
          setUiTheme(res.ui_theme);
        }
      }
    }

    loadSettings();
    return () => {
      active = false;
    };
  }, [refreshSettings, setUiTheme]);

  useEffect(() => {
    if (typeof document === "undefined") return;
    document.documentElement.dataset.uiTheme = uiThemeState;
  }, [uiThemeState]);

  const value = useMemo(() => ({
    socialMode,
    setSocialMode,
    uiTheme: uiThemeState,
    setUiTheme,
    refreshSettings,
  }), [refreshSettings, setUiTheme, socialMode, uiThemeState]);

  return (
    <AppSettingsContext.Provider value={value}>
      {children}
    </AppSettingsContext.Provider>
  );
}

// eslint-disable-next-line react-refresh/only-export-components
export function useAppSettings() {
  return useContext(AppSettingsContext);
}
