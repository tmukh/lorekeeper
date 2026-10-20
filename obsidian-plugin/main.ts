import { App, Notice, Plugin, PluginSettingTab, Setting } from "obsidian";

interface LoreSettings {
  serverUrl: string;
}
const DEFAULT_SETTINGS: LoreSettings = { serverUrl: "http://127.0.0.1:8765" };

export default class LorekeeperPlugin extends Plugin {
  settings: LoreSettings = DEFAULT_SETTINGS;

  async onload() {
    await this.loadSettings();

    // M3 stub: wire these commands to the lorekeeper MCP server (streamable-http).
    // For now they confirm the plugin loads and the server URL is configured.
    this.addCommand({
      id: "lorekeeper-suggest-placement",
      name: "Suggest placement for current note",
      callback: async () => {
        const file = this.app.workspace.getActiveFile();
        if (!file) {
          new Notice("No active note.");
          return;
        }
        new Notice(
          `Lorekeeper: will ask ${this.settings.serverUrl} where "${file.basename}" belongs (M3).`
        );
      },
    });

    this.addCommand({
      id: "lorekeeper-ask",
      name: "Ask the knowledge graph",
      callback: () => new Notice(`Lorekeeper chat panel — coming in M3 (server: ${this.settings.serverUrl}).`),
    });

    this.addSettingTab(new LoreSettingTab(this.app, this));
  }

  async loadSettings() {
    this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
  }
  async saveSettings() {
    await this.saveData(this.settings);
  }
}

class LoreSettingTab extends PluginSettingTab {
  plugin: LorekeeperPlugin;
  constructor(app: App, plugin: LorekeeperPlugin) {
    super(app, plugin);
    this.plugin = plugin;
  }
  display(): void {
    const { containerEl } = this;
    containerEl.empty();
    new Setting(containerEl)
      .setName("Lorekeeper server URL")
      .setDesc("Where the local lorekeeper MCP server is listening (python -m lorekeeper.server --http).")
      .addText((t) =>
        t
          .setPlaceholder("http://127.0.0.1:8765")
          .setValue(this.plugin.settings.serverUrl)
          .onChange(async (v) => {
            this.plugin.settings.serverUrl = v.trim();
            await this.plugin.saveSettings();
          })
      );
  }
}
