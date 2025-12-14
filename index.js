#!/usr/bin/env node

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { WatsonXAI } from "@ibm-cloud/watsonx-ai";
import { IamAuthenticator } from "ibm-cloud-sdk-core";

// Configuration from environment
const WATSONX_API_KEY = process.env.WATSONX_API_KEY;
const WATSONX_PROJECT_ID = process.env.WATSONX_PROJECT_ID;
const WATSONX_URL = process.env.WATSONX_URL || "https://us-south.ml.cloud.ibm.com";

// Initialize watsonx.ai client
let watsonxClient = null;

function getWatsonxClient() {
  if (!watsonxClient && WATSONX_API_KEY) {
    watsonxClient = WatsonXAI.newInstance({
      version: "2024-05-31",
      serviceUrl: WATSONX_URL,
      authenticator: new IamAuthenticator({
        apikey: WATSONX_API_KEY,
      }),
    });
  }
  return watsonxClient;
}

// Create MCP server
const server = new Server(
  {
    name: "watsonx-mcp-server",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// Define available tools
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "watsonx_generate",
        description: "Generate text using IBM watsonx.ai foundation models (Granite, Llama, Mistral, etc.)",
        inputSchema: {
          type: "object",
          properties: {
            prompt: {
              type: "string",
              description: "The prompt to send to the model",
            },
            model_id: {
              type: "string",
              description: "Model ID (e.g., 'ibm/granite-13b-chat-v2', 'meta-llama/llama-3-70b-instruct')",
              default: "ibm/granite-13b-chat-v2",
            },
            max_new_tokens: {
              type: "number",
              description: "Maximum number of tokens to generate",
              default: 500,
            },
            temperature: {
              type: "number",
              description: "Temperature for sampling (0-2)",
              default: 0.7,
            },
            top_p: {
              type: "number",
              description: "Top-p nucleus sampling",
              default: 1.0,
            },
            top_k: {
              type: "number",
              description: "Top-k sampling",
              default: 50,
            },
          },
          required: ["prompt"],
        },
      },
      {
        name: "watsonx_list_models",
        description: "List available foundation models in watsonx.ai",
        inputSchema: {
          type: "object",
          properties: {},
        },
      },
      {
        name: "watsonx_embeddings",
        description: "Generate text embeddings using watsonx.ai embedding models",
        inputSchema: {
          type: "object",
          properties: {
            texts: {
              type: "array",
              items: { type: "string" },
              description: "Array of texts to embed",
            },
            model_id: {
              type: "string",
              description: "Embedding model ID",
              default: "ibm/slate-125m-english-rtrvr",
            },
          },
          required: ["texts"],
        },
      },
      {
        name: "watsonx_chat",
        description: "Have a conversation with watsonx.ai chat models",
        inputSchema: {
          type: "object",
          properties: {
            messages: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  role: { type: "string", enum: ["system", "user", "assistant"] },
                  content: { type: "string" },
                },
              },
              description: "Array of chat messages",
            },
            model_id: {
              type: "string",
              description: "Chat model ID",
              default: "ibm/granite-13b-chat-v2",
            },
            max_new_tokens: {
              type: "number",
              default: 500,
            },
            temperature: {
              type: "number",
              default: 0.7,
            },
          },
          required: ["messages"],
        },
      },
    ],
  };
});

// Handle tool calls
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  const client = getWatsonxClient();

  if (!client) {
    return {
      content: [
        {
          type: "text",
          text: "Error: watsonx.ai not configured. Set WATSONX_API_KEY environment variable.",
        },
      ],
    };
  }

  try {
    switch (name) {
      case "watsonx_generate": {
        const params = {
          input: args.prompt,
          modelId: args.model_id || "ibm/granite-13b-chat-v2",
          parameters: {
            max_new_tokens: args.max_new_tokens || 500,
            temperature: args.temperature || 0.7,
            top_p: args.top_p || 1.0,
            top_k: args.top_k || 50,
          },
        };

        // Add projectId if configured
        if (WATSONX_PROJECT_ID) {
          params.projectId = WATSONX_PROJECT_ID;
        }

        const response = await client.generateText(params);

        const generatedText = response.result.results?.[0]?.generated_text || "";
        return {
          content: [
            {
              type: "text",
              text: generatedText,
            },
          ],
        };
      }

      case "watsonx_list_models": {
        const response = await client.listFoundationModelSpecs({
          limit: 100,
        });

        const models = response.result.resources?.map((m) => ({
          id: m.model_id,
          name: m.label,
          provider: m.provider,
          tasks: m.tasks,
        })) || [];

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(models, null, 2),
            },
          ],
        };
      }

      case "watsonx_embeddings": {
        const params = {
          inputs: args.texts,
          modelId: args.model_id || "ibm/slate-125m-english-rtrvr",
        };

        if (WATSONX_PROJECT_ID) {
          params.projectId = WATSONX_PROJECT_ID;
        }

        const response = await client.embedText(params);

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(response.result, null, 2),
            },
          ],
        };
      }

      case "watsonx_chat": {
        // Format messages for chat completion
        const formattedPrompt = args.messages
          .map((m) => {
            if (m.role === "system") return `System: ${m.content}`;
            if (m.role === "user") return `User: ${m.content}`;
            if (m.role === "assistant") return `Assistant: ${m.content}`;
            return m.content;
          })
          .join("\n\n");

        const params = {
          input: formattedPrompt + "\n\nAssistant:",
          modelId: args.model_id || "ibm/granite-13b-chat-v2",
          parameters: {
            max_new_tokens: args.max_new_tokens || 500,
            temperature: args.temperature || 0.7,
            stop_sequences: ["User:", "System:"],
          },
        };

        if (WATSONX_PROJECT_ID) {
          params.projectId = WATSONX_PROJECT_ID;
        }

        const response = await client.generateText(params);

        const generatedText = response.result.results?.[0]?.generated_text || "";
        return {
          content: [
            {
              type: "text",
              text: generatedText.trim(),
            },
          ],
        };
      }

      default:
        return {
          content: [
            {
              type: "text",
              text: `Unknown tool: ${name}`,
            },
          ],
        };
    }
  } catch (error) {
    return {
      content: [
        {
          type: "text",
          text: `Error calling watsonx.ai: ${error.message}`,
        },
      ],
    };
  }
});

// Start server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("watsonx MCP server running on stdio");
}

main().catch(console.error);
