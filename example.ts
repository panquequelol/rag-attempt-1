import fs from "node:fs/promises";

import {
  Document,
  MetadataMode,
  NodeWithScore,
  VectorStoreIndex,
  Settings,
  Ollama,
  OllamaEmbedding,
  LlamaParseReader,
  MarkdownReader,
  TextQaPrompt,
  ResponseSynthesizer,
  CompactAndRefine,
  OpenAI,
  OpenAIEmbedding,
} from "llamaindex";
import dotenv from "dotenv";
dotenv.config();

async function main() {
  const path = "files/uct_artes.md";
  // const essay = await fs.readFile(path, "utf-8");

  // Settings.llm = new Ollama({
  //   model: "gemma:2b",
  // });
  Settings.llm = new OpenAI({
    model: "gpt-4o-mini",
    apiKey: process.env.OPENAI_API_KEY,
  });
  Settings.embedModel = new OllamaEmbedding({
    model: "mxbai-embed-large",
  });

  const prompt: TextQaPrompt = ({ context, query }) => {
    return `La información del contexto es la siguiente.
  ---------------------
  ${context}
  ---------------------
  Dada la información del contexto y sin conocimientos previos, responde a la consulta.
  Consulta: ${query}
  Respuesta:`;
  };
  const responseSynthesizer = new ResponseSynthesizer({
    responseBuilder: new CompactAndRefine(undefined, prompt),
  });

  // Create Document object with essay
  // const document = new Document({ text: essay, id_: path });
  const reader = new MarkdownReader();
  // const reader = new LlamaParseReader({
  //   language: "es",
  //   resultType: "markdown",
  //   parsingInstruction:
  //     "This PDF contains information about a college major, they are split in semesters. Try to associate the semester and it's corresponding courses, this correspond to the provided major. Ignore links and buttons that let to other sites.",
  //   apiKey: process.env.LLAMAPARSE_API_KEY,
  // });
  const documents = await reader.loadData(path);

  // Split text and create embeddings. Store them in a VectorStoreIndex
  const index = await VectorStoreIndex.fromDocuments(documents);

  // Query the index
  const queryEngine = index.asQueryEngine({
    responseSynthesizer,
  });
  const { response, sourceNodes } = await queryEngine.query({
    query: "La carrera de artes tiene materias de religion en la UCT?",
  });

  // Output response with sources
  console.log("---------------------");
  console.log("La carrera de artes tiene materias de religion en la UCT?");
  console.log("---------------------");
  console.log(response);
  console.log("---------------------");
  if (sourceNodes) {
    sourceNodes.forEach((source: NodeWithScore, index: number) => {
      console.log(
        `\n${index}: Score: ${source.score} - ${source.node
          .getContent(MetadataMode.NONE)
          .substring(0, 50)}...\n`
      );
    });
  }
}

main().catch(console.error);
