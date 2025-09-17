<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import type { OverlayProxy } from "@embedding-atlas/component";
  import type { SearchResultItem } from "./search.js";

  interface Props {
    items: SearchResultItem[];
    highlightItem?: SearchResultItem | null;
    proxy: OverlayProxy;
    nearestNeighborMode?: boolean;
    selectedPointId?: any;
    playerColors?: Map<string, string>;
    playerGroups?: Map<string, SearchResultItem[]>;
  }

  let { items, highlightItem, proxy, nearestNeighborMode = false, selectedPointId, playerColors, playerGroups }: Props = $props();
  
  // Extract player name from item; prefer explicit player_name field
  function getPlayerName(item: SearchResultItem): string {
    const playerFromField = (item as any).fields?.player_name as string | undefined;
    if (playerFromField && playerFromField.trim().length > 0) {
      return playerFromField.trim();
    }
    if (item.text) {
      const match = item.text.match(/^([^,]+)/);
      return (match ? match[1] : item.text.split(' ').slice(0, 2).join(' ')).trim();
    }
    return (String(item.id).split('_')[0] || String(item.id)).trim();
  }
  
  // Get color for a specific item based on player
  function getPlayerColor(item: SearchResultItem, isSelected: boolean): string {
    if (isSelected) {
      return "rgb(239, 68, 68)"; // red-500 for selected point
    }
    
    if (nearestNeighborMode && playerColors) {
      const playerName = getPlayerName(item);
      return playerColors.get(playerName) || "rgb(99, 102, 241)"; // default to indigo
    }
    
    // Fallback to old logic for non-nearest-neighbor mode
    const hue = 200 + (Math.random() * 120); // Blue to green range
    const saturation = 70;
    const lightness = 45;
    return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
  }
  
  // Generate distinct colors for different distance groups (fallback)
  function getDistanceColor(index: number, isSelected: boolean): string {
    if (isSelected) {
      return "rgb(239, 68, 68)"; // red-500 for selected point
    }
    
    // Generate colors in a gradient from blue to green for nearest neighbors
    const hue = 200 + (index * 30) % 120; // Blue to green range
    const saturation = 70 + (index * 10) % 30; // Varying saturation
    const lightness = 45 + (index * 5) % 20; // Varying lightness
    return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
  }
  
  function getPointSize(index: number, isSelected: boolean): number {
    if (isSelected) return 8; // Larger for selected point
    return Math.max(3, 6 - Math.floor(index / 5)); // Decreasing size for farther neighbors
  }
</script>

<svg width={proxy.width} height={proxy.height}>
  <g>
    {#if nearestNeighborMode}
      {#each items as item, index}
        {#if item.x != null && item.y != null}
          {@const loc = proxy.location(item.x, item.y)}
          {@const isSelected = item.id == selectedPointId}
          {@const isHighlight = item.id == highlightItem?.id}
          {@const color = getPlayerColor(item, isSelected)}
          {@const size = getPointSize(index, isSelected)}
          
          {#if isHighlight || isSelected}
            <!-- Cross hair for highlighted/selected point -->
            <line x1={loc.x - 20} x2={loc.x - 10} y1={loc.y} y2={loc.y} stroke={color} stroke-width="2" />
            <line x1={loc.x + 20} x2={loc.x + 10} y1={loc.y} y2={loc.y} stroke={color} stroke-width="2" />
            <line x1={loc.x} x2={loc.x} y1={loc.y - 20} y2={loc.y - 10} stroke={color} stroke-width="2" />
            <line x1={loc.x} x2={loc.x} y1={loc.y + 20} y2={loc.y + 10} stroke={color} stroke-width="2" />
          {/if}
          
          {#if isSelected}
            <!-- Pulsing ring for selected point -->
            <circle cx={loc.x} cy={loc.y} r={size + 4} fill="none" stroke={color} stroke-width="2" opacity="0.6">
              <animate attributeName="r" values="{size + 2};{size + 8};{size + 2}" dur="2s" repeatCount="indefinite" />
              <animate attributeName="opacity" values="0.8;0.2;0.8" dur="2s" repeatCount="indefinite" />
            </circle>
          {/if}
          
          <!-- Main point -->
          <circle 
            cx={loc.x} 
            cy={loc.y} 
            r={size} 
            fill={color} 
            stroke="white" 
            stroke-width={isSelected ? "3" : "2"} 
            opacity={isSelected ? "1" : "0.9"}
          />
          
          {#if !isSelected}
            <!-- Connection line to selected point for nearest neighbors -->
            {#if highlightItem && highlightItem.x != null && highlightItem.y != null}
              {@const selectedLoc = proxy.location(highlightItem.x, highlightItem.y)}
              <line 
                x1={loc.x} 
                y1={loc.y} 
                x2={selectedLoc.x} 
                y2={selectedLoc.y} 
                stroke={color} 
                stroke-width="1" 
                opacity="0.3"
                stroke-dasharray="2,2"
              />
            {/if}
          {/if}
        {/if}
      {/each}
    {:else}
      <!-- Original search result mode -->
      {#each items as item}
        {#if item.x != null && item.y != null}
          {@const loc = proxy.location(item.x, item.y)}
          {@const isHighlight = item.id == highlightItem?.id}
          {#if isHighlight}
            <line x1={loc.x - 20} x2={loc.x - 10} y1={loc.y} y2={loc.y} class="stroke-orange-500" />
            <line x1={loc.x + 20} x2={loc.x + 10} y1={loc.y} y2={loc.y} class="stroke-orange-500" />
            <line x1={loc.x} x2={loc.x} y1={loc.y - 20} y2={loc.y - 10} class="stroke-orange-500" />
            <line x1={loc.x} x2={loc.x} y1={loc.y + 20} y2={loc.y + 10} class="stroke-orange-500" />
          {/if}
          <circle cx={loc.x} cy={loc.y} r={4} class="fill-orange-500 stroke-orange-700 stroke-2" />
        {/if}
      {/each}
    {/if}
  </g>
</svg>
