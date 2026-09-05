import { app } from "../../scripts/app.js";
import {
  LEGACY_ADVANCED_WIDGET_NAMES,
  migrateLegacyAdvancedWidgetValues,
} from "./widget_migration_core.mjs";

app.registerExtension({
  name: "xmarre.VDNH3.LegacyAdvancedWidgetMigration",

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "ApplyVDNH3Advanced") return;

    // Native frontend compatibility metadata for workflows that predate
    // widgets_values_named and used the original 14-widget order.
    nodeData.fallbackWidgetsValuesNames = [...LEGACY_ADVANCED_WIDGET_NAMES];

    const originalConfigure = nodeType.prototype.configure;
    nodeType.prototype.configure = function vdnConfigure(info) {
      migrateLegacyAdvancedWidgetValues(info);
      return originalConfigure.apply(this, arguments);
    };
  },
});
