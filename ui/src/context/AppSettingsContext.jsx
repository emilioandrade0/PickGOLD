import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { getPublicAppSettings } from "../services/auth.js";

const AppSettingsContext = createContext({
  socialMode: false,
  setSocialMode: () => {},
  refreshSettings: async () => ({ ok: false }),
});

export function AppSettingsProvider({ children }) {
  const [socialMode, setSocialMode] = useState(false);

  const refreshSettings = useCallback(async () => {
    const res = await getPublicAppSettings();
    if (res?.ok) {
      setSocialMode(Boolean(res.social_mode));
    }
    return res;
  }, []);

  useEffect(() => {
    let active = true;

    async function loadSettings() {
      const res = await getPublicAppSettings();
      if (active && res?.ok) {
        setSocialMode(Boolean(res.social_mode));
      }
    }

    loadSettings();
    return () => {
      active = false;
    };
  }, [refreshSettings]);

  const value = useMemo(() => ({
    socialMode,
    setSocialMode,
    refreshSettings,
  }), [refreshSettings, socialMode]);

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
