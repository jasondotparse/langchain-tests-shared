import { ChatOpenAI } from "langchain/chat_models/openai";
import {
  LLMSingleActionAgent,
  AgentActionOutputParser,
  AgentExecutor,
} from "langchain/agents";
import { LLMChain } from "langchain/chains";
import {
  renderTemplate,
  BaseChatPromptTemplate,
} from "langchain/prompts";
import { ChainTool } from "langchain/tools";
import { AWSLambda } from "langchain/tools/aws_lambda";
import { S3Loader } from "langchain/document_loaders/web/s3";
import { CharacterTextSplitter } from "langchain/text_splitter";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { RetrievalQAChain } from "langchain/chains";
import { HumanChatMessage } from "langchain/schema";

process.env.LANGCHAIN_HANDLER = "langchain";

const PREFIX = `Complete the following task. You have access to the following tools:`;

const formatInstructions = (toolNames) => `Use the following format:

Task: the task to complete
Thought: you should always think about what to do
Action: the action to take, should be one of [${toolNames}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: The task is now complete.
Task Summary: a short summary of how the task was completed

Note that every Thought should be followed by an Action and Action Input.`;

const SUFFIX = `Begin!

Task: {input}
Thought:{agent_scratchpad}`;

class CustomPromptTemplate extends BaseChatPromptTemplate {
  constructor(args) {
    super({ inputVariables: args.inputVariables });
    this.tools = args.tools;
  }

  _getPromptType() {
    throw new Error("Not implemented");
  }

  async formatMessages(values) {
    const toolStrings = this.tools
      .map((tool) => `${tool.name}: ${tool.description}`)
      .join("\n");
    const toolNames = this.tools.map((tool) => tool.name).join("\n");
    const instructions = formatInstructions(toolNames);
    const template = [PREFIX, toolStrings, instructions, SUFFIX].join("\n\n");
    const intermediateSteps = values.intermediate_steps;
    const agentScratchpad = intermediateSteps.reduce(
      (thoughts, { action, observation }) =>
        thoughts +
        [action.log, `\nObservation: ${observation}`, "Thought:"].join("\n"),
      ""
    );
    const newInput = { agent_scratchpad: agentScratchpad, ...values };
    const formatted = renderTemplate(template, "f-string", newInput);
    return [new HumanChatMessage(formatted)];
  }

  partial(_values) {
    throw new Error("Not implemented");
  }

  serialize() {
    throw new Error("Not implemented");
  }
}

class CustomOutputParser extends AgentActionOutputParser {
  async parse(text) {

    if (text.includes("Task Summary:")) {
      const parts = text.split("Task Summary:");
      const input = parts[parts.length - 1].trim();
      const finalAnswers = { output: input };
      return { log: text, returnValues: finalAnswers };
    }

    const match = /Action: (.*)\nAction Input: (.*)/s.exec(text);
    if (!match) {
      throw new Error(`Could not parse LLM output: ${text}`);
    }

    return {
      tool: match[1].trim(),
      toolInput: match[2].trim().replace(/^"+|"+$/g, ""),
      log: text,
    };
  }

  getFormatInstructions() {
    throw new Error("Not implemented");
  }
}

export const run = async () => {
  const model = new ChatOpenAI({ 
    modelName: 'gpt-4',
    openAIApiKey: '', 
    temperature: 0
  });

  const loader = new S3Loader({
    bucket: "test-bucket",
    key: "minutes.pdf",
    unstructuredAPIURL: "http://localhost:8001/general/v0/general"
  });

  const docs = await loader.load();

  const splitter = new CharacterTextSplitter({
    chunkSize: 750,
    chunkOverlap: 250,
  });

  const splitDocs = await splitter.splitDocuments(docs);

  console.log(splitDocs);

  const vectorStore = await HNSWLib.fromDocuments(splitDocs, new OpenAIEmbeddings({
    openAIApiKey: ""
  }));

  const vectorStoreAsRetriever = vectorStore.asRetriever();

  const chain = RetrievalQAChain.fromLLM(model, vectorStoreAsRetriever);

  const tools = [
    new ChainTool({
      name: "meeting-information-QA-utilty",
      description: "answers questions about the meeting minutes for the most recent meeting at Wigit, LLC",
      chain,
    }),
    new AWSLambda({
      name: "email-sender",
      description: "Sends an email to a single specified email address. Accepts a JSON string with the following keys: to, subject, and body",
      region: "us-east-1",
      accessKeyId: "",
      secretAccessKey: "",
      functionName: "SendEmailViaSES",
    }),
  ];

  const llmChain = new LLMChain({
    prompt: new CustomPromptTemplate({
      tools,
      inputVariables: ["input", "agent_scratchpad"],
    }),
    llm: model,
  });

  const agent = new LLMSingleActionAgent({
    llmChain,
    outputParser: new CustomOutputParser(),
    stop: ["\nObservation"],
  });

  const executor = new AgentExecutor({
    agent,
    tools,
  });

  const input = `Read the meetings notes from the most recent meeting at Wigit, LLC. Email any follow up items to the parties involved.`;

  const result = await executor.call({ input });

  console.log(`Got output: ${result.output}`);
};

run();