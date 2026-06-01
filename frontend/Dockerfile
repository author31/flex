# React + Vite dev image. Source bind-mounted; node_modules kept in-container.
FROM node:22-slim

WORKDIR /app

# Install deps first for layer caching
COPY package.json package-lock.json* ./
RUN npm install

EXPOSE 5173
# Overridden by compose; vite dev server with HMR
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]
