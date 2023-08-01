import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';
// 通用品种
const CONDENSE_PROMPT= `As an AI assistant,use the following conversation and a follow up question, rephrase the follow up question to be a standalone question with chinese.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;

let QA_PROMPT = ``;

console.log('QA_PROMPT:',QA_PROMPT);

export const makeChain = (vectorstore: PineconeStore,pdfNameSpace?:String) => {
  const model = new OpenAI({
    temperature: 0, // increase temepreature to get more creative answers
    modelName: 'gpt-3.5-turbo-0613', //change this to gpt-4 if you have access
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(3),
    {
      qaTemplate: QA_PROMPT,
      questionGeneratorTemplate: CONDENSE_PROMPT,
      returnSourceDocuments: false, //The number of source documents returned is 4 by default
    },
  );
  return chain;
};
