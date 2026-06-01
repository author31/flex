import js from "@eslint/js";

export default [
  { ignores: ["dist/", "node_modules/", "coverage/"] },
  js.configs.recommended,
  {
    files: ["**/*.{ts,tsx}"],
    languageOptions: { ecmaVersion: 2021, sourceType: "module" },
    rules: {},
  },
];
