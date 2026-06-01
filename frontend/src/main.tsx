import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { EditorPage } from "./pages/EditorPage";
import "./styles/theme.css";

const queryClient = new QueryClient();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <EditorPage />
    </QueryClientProvider>
  </StrictMode>,
);
