# Use the official Node.js runtime as base image
FROM node:18-alpine

# Set the working directory inside the container
WORKDIR /app

# Copy package.json and package-lock.json from my-app directory
# This is done before copying the rest of the code to leverage Docker layer caching
COPY my-app/package*.json ./

# Install dependencies
RUN npm ci --omit=dev && npm cache clean --force

# Copy the rest of the application code from my-app directory
COPY my-app/ .

# Create a non-root user for security
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nextjs -u 1001

# Change ownership of the app directory to the nextjs user
RUN chown -R nextjs:nodejs /app
USER nextjs

# Expose the port the app runs on
EXPOSE 3000

# Define the command to run the application
CMD ["npm", "start"]