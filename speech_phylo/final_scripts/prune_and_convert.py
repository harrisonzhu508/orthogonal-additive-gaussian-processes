#!/usr/bin/env python3
"""
Prune a phylogenetic tree to match taxa in a reference tree and convert to BEAST format.

Usage:
    python prune_trees_v3.py <source_tree> <reference_tree> <output_path>
    
Example:
    python prune_trees_v3.py raw_CCD.nex long_v3_CCD0_0_25.nex pruned_beast.nex

If no arguments provided, uses default paths for speech_phylo project.
"""

import re
import sys
from pathlib import Path

import dendropy


def load_tree(filepath: str) -> dendropy.Tree:
    """Load a tree from NEXUS file, preserving BEAST-style annotations."""
    return dendropy.Tree.get(
        path=filepath,
        schema="nexus",
        preserve_underscores=True,
        extract_comment_metadata=True,  # keep [&...] as annotations
    )


def get_taxa(tree: dendropy.Tree) -> set[str]:
    """Extract taxon labels from tree."""
    return {leaf.taxon.label for leaf in tree.leaf_node_iter()}


def prune_tree(tree: dendropy.Tree, keep_taxa: set[str]) -> dendropy.Tree:
    """
    Prune tree to keep only specified taxa.
    Creates a fresh TaxonNamespace to avoid stale references.
    """
    tree.retain_taxa_with_labels(keep_taxa)
    
    # Rebuild TaxonNamespace with only retained taxa
    new_tns = dendropy.TaxonNamespace()
    for leaf in tree.leaf_node_iter():
        new_taxon = new_tns.new_taxon(label=leaf.taxon.label)
        leaf.taxon = new_taxon
    tree.taxon_namespace = new_tns
    
    return tree


def tree_to_newick_with_annotations(tree: dendropy.Tree) -> str:
    """
    Export tree to Newick string preserving BEAST annotations.

    We also strip the final ';' because the NEXUS writer will add it.
    """
    s = tree.as_string(
        schema="newick",
        suppress_annotations=False,   # KEEP [&...] metadata
        suppress_rooting=False,
    ).strip()
    if s.endswith(";"):
        s = s[:-1]
    return s


def create_beast_nexus(taxa: list[str], tree_str: str) -> str:
    """
    Create BEAST-style NEXUS file with Translate block.
    
    Args:
        taxa: List of taxon names (will be sorted)
        tree_str: Newick tree string with taxon names (no trailing ';')
        
    Returns:
        Complete NEXUS file content
    """
    sorted_taxa = sorted(taxa)
    name_to_num = {name: str(i) for i, name in enumerate(sorted_taxa, 1)}
    
    # Taxa block
    taxa_block = "\n".join([f"\t\t\t{t}" for t in sorted_taxa])
    
    # Translate block
    translate_lines = ["\tTranslate"]
    for i, taxon in enumerate(sorted_taxa, 1):
        comma = "," if i < len(sorted_taxa) else ""
        translate_lines.append(f"\t\t{i:>6} {taxon}{comma}")
    translate_lines.append("\t\t\t;")
    translate_block = "\n".join(translate_lines)
    
    # Replace taxon names with numbers in tree string
    # Sort by length (longest first) to avoid partial replacements
    numbered_tree = tree_str
    for name in sorted(name_to_num.keys(), key=len, reverse=True):
        num = name_to_num[name]
        # Match taxon name followed by :, [, ), or , (branch length, annotation, or subtree punctuation)
        pattern = rf'{re.escape(name)}(?=[\[:,);])'
        numbered_tree = re.sub(pattern, num, numbered_tree)
    
    nexus = f"""#NEXUS

Begin taxa;
\tDimensions ntax={len(taxa)};
\t\tTaxlabels
{taxa_block}
\t\t\t;
End;
Begin trees;
{translate_block}
tree TREE_CCD0_CA = {numbered_tree};
End;
"""
    return nexus


def prune_and_convert(
    source_path: str,
    reference_path: str,
    output_path: str,
    verbose: bool = True
) -> dict:
    """
    Prune source tree to match reference taxa and save in BEAST format.
    
    Args:
        source_path: Path to source tree (will be pruned)
        reference_path: Path to reference tree (defines which taxa to keep)
        output_path: Path for output BEAST-format NEXUS file
        verbose: Print progress messages
        
    Returns:
        Dict with statistics about the operation
    """
    if verbose:
        print(f"Loading reference tree: {reference_path}")
    ref_tree = load_tree(reference_path)
    target_taxa = get_taxa(ref_tree)
    
    if verbose:
        print(f"Loading source tree: {source_path}")
    source_tree = load_tree(source_path)
    source_taxa = get_taxa(source_tree)
    
    # Check overlap
    overlap = target_taxa & source_taxa
    missing = target_taxa - source_taxa
    extra = source_taxa - target_taxa
    
    if verbose:
        print(f"\nSource taxa: {len(source_taxa)}")
        print(f"Target taxa: {len(target_taxa)}")
        print(f"Overlap: {len(overlap)}")
        
        if missing:
            print(f"\n⚠️  Missing from source ({len(missing)}):")
            for t in sorted(missing):
                print(f"    {t}")
    
    if not overlap:
        raise ValueError("No overlapping taxa between source and reference trees")
    
    # Prune
    if verbose:
        print(f"\nPruning {len(extra)} taxa...")
    pruned_tree = prune_tree(source_tree, overlap)
    
    # Verify
    final_taxa = list(get_taxa(pruned_tree))
    if verbose:
        print(f"Final taxa: {len(final_taxa)}")
        print(f"TaxonNamespace size: {len(pruned_tree.taxon_namespace)}")
    
    # Get tree string with annotations
    tree_str = tree_to_newick_with_annotations(pruned_tree)
    
    # Convert to BEAST format
    if verbose:
        print("\nConverting to BEAST format...")
    beast_nexus = create_beast_nexus(final_taxa, tree_str)
    
    # Save BEAST format
    if verbose:
        print(f"Saving to: {output_path}")
    Path(output_path).write_text(beast_nexus)
    
    # Also save Newick (annotated)
    newick_path = str(output_path).replace(".nex", ".newick")
    if newick_path != output_path:
        Path(newick_path).write_text(tree_str + ";\n")
        if verbose:
            print(f"Also saved: {newick_path}")
    
    stats = {
        "source_taxa": len(source_taxa),
        "target_taxa": len(target_taxa),
        "overlap": len(overlap),
        "missing": list(missing),
        "pruned": len(extra),
        "final_taxa": sorted(final_taxa),
    }
    
    if verbose:
        print(f"\n{'='*50}")
        print("FINAL TAXA")
        print("="*50)
        for t in sorted(final_taxa):
            print(f"  {t}")
    
    return stats


def main():
    # Default paths for speech_phylo project
    default_source = "speech_phylo/heggarty2024_raw.nex"
    default_reference = "speech_phylo/long_v3_CCD0.0.25.nex"
    default_output = "speech_phylo/heggarty2024.nex"
    
    if len(sys.argv) >= 4:
        source_path = sys.argv[1]
        reference_path = sys.argv[2]
        output_path = sys.argv[3]
    elif len(sys.argv) == 1:
        # Use defaults
        print("Using default paths...")
        source_path = default_source
        reference_path = default_reference
        output_path = default_output
    else:
        print(__doc__)
        print("\nError: Provide all 3 arguments or none (for defaults)")
        sys.exit(1)
    
    # Validate inputs exist
    if not Path(source_path).exists():
        print(f"Error: Source file not found: {source_path}")
        sys.exit(1)
    if not Path(reference_path).exists():
        print(f"Error: Reference file not found: {reference_path}")
        sys.exit(1)
    
    try:
        prune_and_convert(source_path, reference_path, output_path)
        print("\n✓ Done!")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
