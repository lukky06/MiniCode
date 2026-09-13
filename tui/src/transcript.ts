import { VStack, type Component } from "@earendil-works/pi-tui";

import { Activity, type ToolFinishDetails } from "./components/activity.js";
import { AssistantMessage } from "./components/assistant-message.js";
import { ReasoningMessage } from "./components/reasoning-message.js";
import { UserMessage } from "./components/user-message.js";

export class Transcript implements Component {
  private readonly stack = new VStack([], { gap: 1 });
  private activity: Activity | null = null;
  private assistant: AssistantMessage | null = null;
  private reasoning: ReasoningMessage | null = null;
  private readonly reasoningMessages: ReasoningMessage[] = [];
  private reasoningExpanded = false;

  appendUser(text: string): void {
    this.completeReasoning();
    this.stack.addChild(new UserMessage(text));
    this.assistant = null;
  }

  appendReasoningDelta(text: string): void {
    if (!text) return;
    if (this.reasoning === null) {
      if (this.activity !== null) {
        this.activity.complete();
        this.activity = null;
      }
      this.reasoning = new ReasoningMessage();
      this.reasoning.setExpanded(this.reasoningExpanded);
      this.reasoningMessages.push(this.reasoning);
      this.stack.addChild(this.reasoning);
    }
    this.reasoning.appendDelta(text);
  }

  completeReasoning(): void {
    this.reasoning?.finish();
    this.reasoning = null;
  }

  toggleReasoning(): boolean {
    this.reasoningExpanded = !this.reasoningExpanded;
    for (const reasoning of this.reasoningMessages) {
      reasoning.setExpanded(this.reasoningExpanded);
    }
    return this.reasoningExpanded;
  }

  startTool(id: string, action: string, target?: string | null): void {
    this.completeReasoning();
    if (this.activity === null) {
      this.activity = new Activity();
      this.stack.addChild(this.activity);
    }
    this.activity.start(id, action, target);
  }

  finishTool(
    id: string,
    status: string,
    summary?: string | null,
    details: ToolFinishDetails = {},
  ): void {
    this.activity?.finish(id, status, summary, details);
  }

  completeActivity(): void {
    this.activity?.complete();
  }

  appendAssistantDelta(text: string): void {
    this.completeReasoning();
    if (this.assistant === null) {
      this.assistant = new AssistantMessage();
      this.stack.addChild(this.assistant);
    }
    this.assistant.appendDelta(text);
  }

  startRun(): void {
    this.completeReasoning();
    this.activity = null;
    this.assistant = null;
  }

  get currentAssistantText(): string {
    return this.assistant?.text ?? "";
  }

  invalidate(): void {
    this.stack.invalidate();
  }

  render(width: number): string[] {
    return this.stack.render(width);
  }
}
