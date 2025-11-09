import './tracerSetup';
import { Agent } from '@mastra/core/agent';
import { z } from 'zod';
import { mapMastraTools, withMastraTracing } from '@fledgling/tracer';

import { registerAgentWithBackend } from './registerAgent';

const CUSTOMER_SERVICE_AGENT_ID = process.env.LANGFUSE_CUSTOMER_SERVICE_AGENT_ID || 'customer-service-agent';

// Mock Order Lookup Tool
async function lookupOrder(orderId: string) {
  // Mock implementation
  return {
    orderId,
    status: 'delivered',
    items: ['Product A', 'Product B'],
    orderDate: '2024-01-15',
    customerEmail: 'customer@example.com',
    totalAmount: 129.99,
  };
}

// Mock Delivery Tracking Tool
async function trackDelivery(trackingNumber: string) {
  // Mock implementation
  return {
    trackingNumber,
    status: 'in_transit',
    currentLocation: 'Distribution Center - Chicago',
    estimatedDelivery: '2024-01-20',
    carrier: 'Express Shipping Co.',
    lastUpdate: '2024-01-18T10:30:00Z',
  };
}

// Mock Delivery Evidence Checker
async function checkDeliveryEvidence(trackingNumber: string) {
  // Mock implementation
  return {
    trackingNumber,
    hasPhoto: true,
    hasSignature: true,
    deliveryProof: {
      photoUrl: 'https://example.com/delivery-photo.jpg',
      signature: 'John Doe',
      deliveredAt: '2024-01-18T14:22:00Z',
      deliveryAddress: '123 Main St, City, State 12345',
    },
    verified: true,
  };
}

// Mock Refund Policy Engine
async function checkRefundPolicy(orderId: string, reason: string) {
  // Mock implementation
  const eligible = reason.toLowerCase().includes('defective') || reason.toLowerCase().includes('wrong item');
  return {
    orderId,
    eligible,
    refundAmount: eligible ? 129.99 : 0,
    policy: {
      timeLimit: '30 days',
      conditions: ['Unopened items', 'Defective products', 'Wrong items received'],
      processingTime: '5-7 business days',
    },
    reason,
  };
}

// Mock Replacement Policy Engine
async function checkReplacementPolicy(orderId: string, itemName: string) {
  // Mock implementation
  const inStock = Math.random() > 0.3; // 70% chance in stock
  return {
    orderId,
    itemName,
    eligible: true,
    inStock,
    replacementOptions: inStock
      ? [
          { option: 'Same item', available: true, shippingTime: '2-3 business days' },
          { option: 'Upgrade', available: true, shippingTime: '3-5 business days' },
        ]
      : [{ option: 'Same item', available: false, shippingTime: '7-10 business days' }],
    policy: {
      timeLimit: '30 days',
      conditions: ['Defective items', 'Damaged during shipping'],
    },
  };
}

// Mock Case Management System
async function createCase(customerEmail: string, issueType: string, description: string) {
  // Mock implementation
  const caseId = `CASE-${Date.now()}`;
  return {
    caseId,
    customerEmail,
    issueType,
    description,
    status: 'open',
    priority: issueType.toLowerCase().includes('urgent') ? 'high' : 'medium',
    assignedTo: 'Support Team A',
    createdAt: new Date().toISOString(),
    estimatedResolution: '24-48 hours',
  };
}

// Mock Payment Processing API
async function processRefund(orderId: string, amount: number, reason: string) {
  // Mock implementation
  const transactionId = `TXN-${Date.now()}`;
  return {
    transactionId,
    orderId,
    amount,
    reason,
    status: 'processed',
    refundMethod: 'original_payment_method',
    estimatedArrival: '5-7 business days',
    confirmationNumber: `REF-${Date.now()}`,
  };
}

const CUSTOMER_SERVICE_INSTRUCTIONS = `You are an expert customer service orchestrator agent for an online shipping company. Your role is to handle customer service requests including returns, exchanges, order inquiries, delivery tracking, and refunds.

IMPORTANT: Use tools intelligently based on what the customer is asking. Only use the tools that are relevant to the specific request. Do not use multiple tools unless the workflow requires it.

Your workflow should be:
1. If customer mentions an order number/ID or asks about order details → use lookupOrder tool
2. If customer mentions a tracking number or asks "where is my package" → use trackDelivery tool
3. If customer claims they didn't receive a package → use checkDeliveryEvidence tool
4. If customer requests a refund → FIRST use checkRefundPolicy to verify eligibility, THEN use processRefund if eligible
5. If customer wants to replace/exchange an item → use checkReplacementPolicy tool
6. For complex issues needing follow-up → use createCase tool

Available tools (use these tools when relevant):
- lookupOrder: Use this tool when a customer mentions an order number, order ID, or asks about their order status, items, or order information. Do not guess order details - use this tool to get accurate order data.
- trackDelivery: Use this tool when a customer mentions a tracking number, asks "where is my package", "when will it arrive", or wants delivery status information. Do not guess delivery status - use this tool to get real-time tracking data.
- checkDeliveryEvidence: Use this tool when a customer claims they did not receive their package, says "it was never delivered", or needs proof of delivery. This tool provides actual delivery proof data including photos and signatures.
- checkRefundPolicy: Use this tool BEFORE processing any refund request to determine eligibility and refund amount. Never process a refund without checking policy first using this tool.
- checkReplacementPolicy: Use this tool when a customer wants to exchange, replace, or get a new item. This tool provides replacement options, availability, and shipping times.
- createCase: Use this tool for complex issues that require follow-up, escalation, or cannot be resolved immediately. This creates a tracked case with a case ID that the customer can reference.
- processRefund: Use this tool to actually execute refund transactions. Only use AFTER confirming eligibility with checkRefundPolicy. This tool processes the refund and provides a confirmation number.

Be professional, empathetic, and efficient. Use tools to get accurate information, but only use the tools that are relevant to the customer's specific request.`;

const customerServiceTools = {
    lookupOrder: {
      description: 'Look up order details by order ID. Use this tool when a customer mentions an order number, order ID, or asks about their order status, items, or order information. Do not guess order details - use this tool to get accurate order data.',
      parameters: z.object({
        orderId: z.string().describe('The order ID or order number provided by the customer'),
      }),
      execute: async ({ orderId }: { orderId: string }) => {
        return await lookupOrder(orderId);
      },
    },
    trackDelivery: {
      description: 'Track the delivery status of a shipment using a tracking number. Use this tool when a customer mentions a tracking number, asks "where is my package", "when will it arrive", or wants delivery status information. Do not guess delivery status - use this tool to get real-time tracking data.',
      parameters: z.object({
        trackingNumber: z.string().describe('The tracking number for the shipment'),
      }),
      execute: async ({ trackingNumber }: { trackingNumber: string }) => {
        return await trackDelivery(trackingNumber);
      },
    },
    checkDeliveryEvidence: {
      description: 'Check delivery evidence including photos and signatures. Use this tool when a customer claims they did not receive their package, says "it was never delivered", or needs proof of delivery. This tool provides actual delivery proof data including photos and signatures.',
      parameters: z.object({
        trackingNumber: z.string().describe('The tracking number to check delivery evidence for'),
      }),
      execute: async ({ trackingNumber }: { trackingNumber: string }) => {
        return await checkDeliveryEvidence(trackingNumber);
      },
    },
    checkRefundPolicy: {
      description: 'Check if a customer is eligible for a refund based on the refund policy. Use this tool BEFORE processing any refund request to determine eligibility and refund amount. Never process a refund without checking policy first using this tool.',
      parameters: z.object({
        orderId: z.string().describe('The order ID for the refund request'),
        reason: z.string().describe('The reason the customer is requesting a refund'),
      }),
      execute: async ({ orderId, reason }: { orderId: string; reason: string }) => {
        return await checkRefundPolicy(orderId, reason);
      },
    },
    checkReplacementPolicy: {
      description: 'Check if a customer is eligible for a replacement item and what options are available. Use this tool when a customer wants to exchange, replace, or get a new item. This tool provides replacement options, availability, and shipping times.',
      parameters: z.object({
        orderId: z.string().describe('The order ID for the replacement request'),
        itemName: z.string().describe('The name or description of the item to be replaced'),
      }),
      execute: async ({ orderId, itemName }: { orderId: string; itemName: string }) => {
        return await checkReplacementPolicy(orderId, itemName);
      },
    },
    createCase: {
      description: 'Create a support case in the case management system. Use this tool for complex issues that require follow-up, escalation, or cannot be resolved immediately. This creates a tracked case with a case ID that the customer can reference.',
      parameters: z.object({
        customerEmail: z.string().email().describe('The customer email address'),
        issueType: z.string().describe('The type of issue (e.g., "return request", "delivery issue", "refund request")'),
        description: z.string().describe('A detailed description of the customer issue'),
      }),
      execute: async ({ customerEmail, issueType, description }: { customerEmail: string; issueType: string; description: string }) => {
        return await createCase(customerEmail, issueType, description);
      },
    },
    processRefund: {
      description: 'Process a refund payment to the customer. Use this tool to actually execute refund transactions. Only use AFTER confirming eligibility with checkRefundPolicy. This tool processes the refund and provides a confirmation number.',
      parameters: z.object({
        orderId: z.string().describe('The order ID to process refund for'),
        amount: z.number().describe('The refund amount in dollars'),
        reason: z.string().describe('The reason for the refund'),
      }),
      execute: async ({ orderId, amount, reason }: { orderId: string; amount: number; reason: string }) => {
        return await processRefund(orderId, amount, reason);
      },
    },
} as const;

const rawCustomerServiceAgent = new Agent({
  name: 'customer-service-orchestrator',
  instructions: CUSTOMER_SERVICE_INSTRUCTIONS,
  model: 'openai/gpt-4o',
  tools: customerServiceTools,
});

// Wrap the agent so all runs, LLM generations, and tool calls automatically log to Langfuse.
export const qaAgent = withMastraTracing(rawCustomerServiceAgent, {
  agentId: CUSTOMER_SERVICE_AGENT_ID,
  tags: ['customer-service', 'orchestrator', 'shipping'],
});

setTimeout(() => {
  console.log(`[Agent Setup] Registering customer service agent: ${CUSTOMER_SERVICE_AGENT_ID}`);
  void registerAgentWithBackend({
    id: CUSTOMER_SERVICE_AGENT_ID,
    name: 'Customer Service Orchestrator',
    taskDescription: 'Handles customer service requests including returns, exchanges, order inquiries, delivery tracking, and refunds',
    instructions: CUSTOMER_SERVICE_INSTRUCTIONS,
    originalLLM: 'openai/gpt-4o',
    tags: ['customer-service', 'orchestrator', 'shipping'],
    tools: mapMastraTools(CUSTOMER_SERVICE_AGENT_ID, customerServiceTools),
  });
}, 5000);
