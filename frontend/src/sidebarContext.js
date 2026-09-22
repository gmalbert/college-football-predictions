/**
 * Sidebar-extra channel.
 *
 * Streamlit pages can append widgets to the sidebar at any point in the script
 * (Preseason Outlook does).  A context keeps that behaviour without making the
 * page components import the app shell.
 */
import { createContext, useContext } from "react";

export const SidebarExtraContext = createContext(() => {});

export const useSidebarExtra = () => useContext(SidebarExtraContext);

/**
 * The ``<h1>`` text for the current route, taken from ``/api/meta``.
 *
 * ``/api/meta`` is a few hundred bytes and resolves in ~8 ms, while a page
 * payload can take much longer.  Knowing the title up front lets the page
 * shell paint immediately instead of waiting on the data request.
 */
export const PageTitleContext = createContext(null);

export const usePageTitle = () => useContext(PageTitleContext);
