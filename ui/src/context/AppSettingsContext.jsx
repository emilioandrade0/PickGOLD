import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { getPublicAppSettings } from "../services/auth.js";

const AppSettingsContext = createContext({
  socialMode: false,
  setSocialMode: () => {},
  uiTheme: "original",
  setUiTheme: () => {},
  classicLightNightMode: false,
  setClassicLightNightMode: () => {},
  refreshSettings: async () => ({ ok: false }),
});

const UI_THEME_STORAGE_KEY = "pickgold_ui_theme";
const CLASSIC_LIGHT_NIGHT_MODE_STORAGE_KEY = "pickgold_classic_light_night_mode";

function normalizeUiTheme(value) {
  const txt = String(value || "").trim().toLowerCase();
  if (txt === "dashboard_pro") return "dashboard_pro";
  if (txt === "classic_light") return "classic_light";
  return "original";
}

function getStoredUiTheme() {
  if (typeof window === "undefined") return "original";
  return normalizeUiTheme(window.localStorage.getItem(UI_THEME_STORAGE_KEY));
}

function normalizeClassicLightNightMode(value) {
  if (value === true || value === false) return value;
  const txt = String(value || "").trim().toLowerCase();
  return txt === "1" || txt === "true" || txt === "yes" || txt === "si" || txt === "on";
}

function getStoredClassicLightNightMode() {
  if (typeof window === "undefined") return false;
  return normalizeClassicLightNightMode(window.localStorage.getItem(CLASSIC_LIGHT_NIGHT_MODE_STORAGE_KEY));
}

export function AppSettingsProvider({ children }) {
  const [socialMode, setSocialMode] = useState(false);
  const [uiThemeState, setUiThemeState] = useState(() => getStoredUiTheme());
  const [classicLightNightModeState, setClassicLightNightModeState] = useState(() => getStoredClassicLightNightMode());

  const setUiTheme = useCallback((value) => {
    const next = normalizeUiTheme(value);
    setUiThemeState(next);
    if (typeof window !== "undefined") {
      window.localStorage.setItem(UI_THEME_STORAGE_KEY, next);
    }
  }, []);

  const setClassicLightNightMode = useCallback((value) => {
    const next = normalizeClassicLightNightMode(value);
    setClassicLightNightModeState(next);
    if (typeof window !== "undefined") {
      window.localStorage.setItem(CLASSIC_LIGHT_NIGHT_MODE_STORAGE_KEY, next ? "1" : "0");
    }
  }, []);

  const refreshSettings = useCallback(async () => {
    const res = await getPublicAppSettings();
    if (res?.ok) {
      setSocialMode(Boolean(res.social_mode));
      if (res.ui_theme) {
        setUiTheme(res.ui_theme);
      }
      setClassicLightNightMode(Boolean(res.classic_light_night_mode));
    }
    return res;
  }, [setClassicLightNightMode, setUiTheme]);

  useEffect(() => {
    let active = true;

    async function loadSettings() {
      const res = await getPublicAppSettings();
      if (active && res?.ok) {
        setSocialMode(Boolean(res.social_mode));
        if (res.ui_theme) {
          setUiTheme(res.ui_theme);
        }
        setClassicLightNightMode(Boolean(res.classic_light_night_mode));
      }
    }

    loadSettings();
    return () => {
      active = false;
    };
  }, [refreshSettings, setClassicLightNightMode, setUiTheme]);

  useEffect(() => {
    if (typeof document === "undefined") return;
    document.documentElement.dataset.uiTheme = uiThemeState;
  }, [uiThemeState]);

  useEffect(() => {
    if (typeof document === "undefined") return;
    document.documentElement.dataset.classicLightNight = classicLightNightModeState ? "1" : "0";
  }, [classicLightNightModeState]);

  const value = useMemo(() => ({
    socialMode,
    setSocialMode,
    uiTheme: uiThemeState,
    setUiTheme,
    classicLightNightMode: classicLightNightModeState,
    setClassicLightNightMode,
    refreshSettings,
  }), [classicLightNightModeState, refreshSettings, setClassicLightNightMode, setUiTheme, socialMode, uiThemeState]);

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
