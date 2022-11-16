import { LitElement, css, html } from "lit";
import { customElement } from "lit/decorators.js";

/**
 * title
 *
 */
@customElement("population-root")
export class PopulationRoot extends LitElement {
  render() {
    return html`
      <top-app-bar title="Population"> </top-app-bar>
      <div class="main-layout">
        <div>Charts and Biomarkers</div>
        <div>Image Viewer</div>
      </div>
    `;
  }
  static styles = css`
    :host {
      height: 100%;
      display: flex;
      flex-direction: column;
    }

    .main-layout {
      flex: 1;
      display: flex;
      flex-direction: column;
    }

    .main-layout > div {
      flex: 1;
    }
  `;
}

declare global {
  interface HTMLElementTagNameMap {
    "population-root": PopulationRoot;
  }
}
