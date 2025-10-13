from __future__ import annotations

from io import StringIO
import json
from typing import Any, Dict, Optional, Tuple, List

import streamlit as st

from ase import Atoms
from ase.io import write as ase_write

from src.theme_config import ThemeConfig






def hash_atoms(atoms: Atoms):
    """Create a stable hash for an ASE Atoms object."""
    # A robust hash can be made from positions, numbers, cell, and pbc.
    # We convert arrays to bytes to ensure consistent hashing.
    positions_bytes = atoms.get_positions().tobytes()
    numbers_bytes = atoms.get_atomic_numbers().tobytes()
    cell_bytes = atoms.get_cell().tobytes()
    pbc_tuple = tuple(atoms.get_pbc())
    return (positions_bytes, numbers_bytes, cell_bytes, pbc_tuple)


SNAPSHOT_QUALITY_OPTIONS: List[Tuple[str, int]] = [
    ("Standard (1x)", 1),
    ("High (2x)", 2),
    ("Ultra (4x)", 4),
]


def sanitize_filename(label: str, suffix: str = ".png") -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label.strip())
    safe = safe or "snapshot"
    return f"{safe}{suffix}"


def _safe_dom_id(value: Optional[str], *, default: str = "viewer") -> str:
    if not value:
        return default
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in value)
    cleaned = cleaned.strip("-")
    return cleaned or default


def filter_hydrogens(atoms: Atoms, *, show_hydrogens: bool) -> Atoms:
    if show_hydrogens:
        return atoms
    symbols = atoms.get_chemical_symbols()
    keep_indices = [idx for idx, symbol in enumerate(symbols) if symbol != "H"]
    if len(keep_indices) == len(symbols):
        return atoms
    if not keep_indices:
        return Atoms()
    return atoms[keep_indices]


def atoms_to_pdb_block(atoms: Atoms) -> str:
    centered = atoms.copy()
    try:
        center = centered.get_center_of_mass()
    except Exception:  # fallback if masses undefined
        center = centered.get_positions().mean(axis=0)
    centered.translate(-center)
    buffer = StringIO()
    ase_write(buffer, centered, format="proteindatabank")
    return buffer.getvalue()


@st.cache_data(hash_funcs={Atoms: hash_atoms})
def render_ngl_view(
    atoms: Atoms,
    label: str,
    *,
    theme: ThemeConfig,
    sphere_radius: float,
    bond_radius: float,
    interaction_mode: str,
    height: int,
    width: int = 700,
    representation_style: str = "Ball + Stick",
    label_mode: Optional[str] = None,
    snapshot: Optional[Dict[str, Any]] = None,
) -> str:
    pdb_block = atoms_to_pdb_block(atoms)
    metadata = [
        {
            "index": idx,
            "serial": idx + 1,
            "symbol": sym,
            "atomic_number": int(num),
            "mass": float(mass),
            "x": float(pos[0]),
            "y": float(pos[1]),
            "z": float(pos[2]),
        }
        for idx, (sym, num, mass, pos) in enumerate(
            zip(
                atoms.get_chemical_symbols(),
                atoms.get_atomic_numbers(),
                atoms.get_masses(),
                atoms.get_positions(),
            )
        )
    ]

    default_quality_options = [
        {"label": label, "value": str(factor)} for label, factor in SNAPSHOT_QUALITY_OPTIONS
    ]
    snapshot_cfg: Dict[str, Any] = {
        "transparent": False,
        "factor": 1.0,
        "antialias": True,
        "trim": True,
        "filename": sanitize_filename(label),
        "background": theme.background,
        "qualityOptions": default_quality_options,
    }
    if snapshot:
        for key, value in snapshot.items():
            if value is None:
                continue
            if key == "quality_options":
                snapshot_cfg["qualityOptions"] = value
                continue
            snapshot_cfg[key] = value
    try:
        snapshot_cfg["factor"] = float(snapshot_cfg.get("factor", 1.0))
    except (TypeError, ValueError):
        snapshot_cfg["factor"] = 1.0

    mode_presets = {
        "rotate": {"label": "Rotate / navigate", "maxAtoms": 0},
        "select": {"label": "Select atom", "maxAtoms": 1},
        "measurement": {"label": "Measurement", "maxAtoms": 4},
    }
    mode_key = interaction_mode if interaction_mode in mode_presets else "rotate"
    mode_cfg = mode_presets[mode_key]

    sphere_radius = max(float(sphere_radius), 0.1)
    bond_radius = max(float(bond_radius), 0.02)
    aspect_ratio = max(sphere_radius / bond_radius if bond_radius else 1.0, 1.0)
    highlight_ratio = max((sphere_radius + 0.2) / bond_radius if bond_radius else aspect_ratio, 1.0)

    label_modes = []
    if label_mode:
        label_modes.append(label_mode.lower().replace(" ", "_"))

    style_map = {
        "Ball + Stick": "ball+stick",
        "Licorice": "licorice",
        "Spacefilling": "spacefill",
        "Hyperball": "hyperball",
        "Line": "line",
        "Point Cloud": "point",
        "Surface": "surface",
    }
    style_key = style_map.get(representation_style, "ball+stick")

    style_params: Dict[str, Any] = {"colorScheme": "element"}
    if style_key == "ball+stick":
        style_params.update(
            {
                "multipleBond": "symmetric",
                "sphereDetail": 2,
                "radiusType": "size",
                "radiusSize": bond_radius,
                "aspectRatio": aspect_ratio,
            }
        )
    elif style_key == "licorice":
        style_params.update(
            {
                "multipleBond": "symmetric",
                "radiusType": "size",
                "radiusSize": bond_radius,
            }
        )
    elif style_key == "spacefill":
        style_params.update(
            {
                "radiusType": "size",
                "radiusSize": sphere_radius,
            }
        )
    elif style_key == "hyperball":
        style_params.update(
            {
                "multipleBond": "symmetric",
                "radius": max(bond_radius, 0.05),
                "shrink": 0.2,
            }
        )
    elif style_key == "line":
        style_params.update(
            {
                "linewidth": max(1, int(round(bond_radius * 30))),
                "multipleBond": "off",
            }
        )
    elif style_key == "point":
        style_params.update(
            {
                "pointSize": max(1.0, sphere_radius * 20.0),
                "alpha": 1.0,
            }
        )
    elif style_key == "surface":
        style_params.update(
            {
                "opacity": 0.85,
                "surfaceType": "msms",
                "probeRadius": 1.4,
                "contour": False,
            }
        )

    payload = {
        "pdb": pdb_block,
        "atoms": metadata,
        "mode": mode_key,
        "modeLabel": mode_cfg["label"],
        "maxAtoms": mode_cfg["maxAtoms"],
        "sphereRadius": sphere_radius,
        "bondRadius": bond_radius,
        "aspectRatio": aspect_ratio,
        "highlightAspectRatio": highlight_ratio,
        "labelModes": label_modes,
        "theme": {
            "background": theme.background,
            "text": theme.text_color,
            "highlight": theme.highlight,
        },
        "palette": ["#FF4136", "#2ECC40", "#0074D9", "#B10DC9", "#FF851B"],
        "style": style_key,
        "styleParams": style_params,
        "zoomStep": 0.18,
        "snapshot": snapshot_cfg,
    }

    # Build the quality options HTML with numeric comparison (fix C)
    quality_options_html = "".join(
        f"<option value=\"{opt['value']}\""
        f"{' selected' if float(snapshot_cfg.get('factor', 1.0)) == float(opt['value']) else ''}>"
        f"{opt['label']}</option>"
        for opt in snapshot_cfg.get("qualityOptions", [])
    )

    # --- Keep everything else above the return the same (payload, etc.) ---

    # Build JS as a *plain* string so braces don't need escaping
    js = """
    (function() {
      var cfg = __PAYLOAD__;
      var stage = new NGL.Stage('ngl-stage', { backgroundColor: cfg.theme.background });
      window.addEventListener('resize', function() { stage.handleResize(); }, false);

      var selection = [];
      var highlightReprs = [];
      var measurementReprs = [];

      function clearReprs(list) {
        (list || []).forEach(function(repr) { try { repr.dispose(); } catch (err) { } });
        list.length = 0;
      }

      function ensureZoomControls() {
        var node = document.getElementById('ngl-stage');
        if (!node) return;
        var controls = node.querySelector('.molviewer-zoom-controls');
        if (controls) return controls;

        controls = document.createElement('div');
        controls.className = 'molviewer-zoom-controls';
        Object.assign(controls.style, {
          position: 'absolute', top: '12px', right: '12px', display: 'flex', gap: '6px',
          pointerEvents: 'auto', zIndex: 20
        });

        function makeButton(label, title, onClick) {
          var btn = document.createElement('button');
          btn.type = 'button';
          btn.textContent = label;
          btn.title = title;
          Object.assign(btn.style, {
            width: '34px', height: '34px', borderRadius: '6px', border: '1px solid rgba(15,23,42,0.2)',
            background: cfg.theme.background, color: cfg.theme.text, cursor: 'pointer',
            fontSize: '18px', lineHeight: '32px', padding: '0', boxShadow: '0 4px 12px rgba(15,23,42,0.12)'
          });
          btn.addEventListener('click', function(ev) {
            ev.preventDefault();
            ev.stopPropagation();
            onClick();
          });
          btn.addEventListener('mouseenter', function() {
            btn.style.background = cfg.theme.highlight;
            btn.style.color = cfg.theme.background;
          });
          btn.addEventListener('mouseleave', function() {
            btn.style.background = cfg.theme.background;
            btn.style.color = cfg.theme.text;
          });
          return btn;
        }

        var step = Math.max(Math.min(cfg.zoomStep || 0.18, 0.5), 0.02);
        var zoomIn = makeButton('+', 'Zoom in', function() { try { stage.viewerControls.zoom(step); } catch (err) { console.warn('Zoom in failed', err); }});
        var zoomOut = makeButton('−', 'Zoom out', function() { try { stage.viewerControls.zoom(-step); } catch (err) { console.warn('Zoom out failed', err); }});
        var reset = makeButton('↻', 'Reset view', function() { try { stage.autoView(); } catch (err) { console.warn('Reset view failed', err); }});

        controls.appendChild(zoomIn);
        controls.appendChild(zoomOut);
        controls.appendChild(reset);
        node.appendChild(controls);

        return controls;
      }

      function enableGestureZoom() {
        if (!stage || !stage.viewer) return;

        if (stage.viewer.container && stage.viewer.container.style.touchAction !== 'none') {
          stage.viewer.container.style.touchAction = 'none';
        }

        if (stage.mouseControls && stage.trackballControls) {
          var hasScrollAction = stage.mouseControls.actionList && stage.mouseControls.actionList.some(function(action) {
            return action && action.type === 'scroll';
          });
          if (!hasScrollAction) {
            stage.mouseControls.add('scroll', function(stageRef, delta) {
              stageRef.trackballControls.zoom(delta);
            });
          }
        }

        var canvas = stage.viewer && stage.viewer.renderer ? stage.viewer.renderer.domElement : null;
        if (!canvas || canvas.dataset.molviewerPinchBound) return;
        canvas.dataset.molviewerPinchBound = '1';

        var pinchState = null;
        function pinchDistance(touches) {
          if (!touches || touches.length < 2) return 0;
          var dx = touches[0].pageX - touches[1].pageX;
          var dy = touches[0].pageY - touches[1].pageY;
          return Math.sqrt(dx * dx + dy * dy);
        }

        canvas.addEventListener('touchstart', function(event) {
          if (event.touches && event.touches.length === 2) {
            pinchState = {
              start: pinchDistance(event.touches) || 0,
              camera: stage.viewerControls ? stage.viewerControls.getCameraDistance() : stage.viewer ? stage.viewer.cameraDistance : 0
            };
          }
        }, { passive: true });

        canvas.addEventListener('touchmove', function(event) {
          if (!pinchState || !event.touches || event.touches.length !== 2) return;
          event.preventDefault();
          var distance = pinchDistance(event.touches);
          if (!distance || !pinchState.start) return;
          var ratio = pinchState.start / distance;
          var target = pinchState.camera * ratio;
          if (stage.viewerControls && typeof stage.viewerControls.distance === 'function') {
            stage.viewerControls.distance(target);
          } else if (stage.viewer) {
            stage.viewer.cameraDistance = Math.max(Math.abs(target), 0.2);
            stage.viewer.updateZoom();
          }
        }, { passive: false });

        canvas.addEventListener('touchend', function(event) {
          if (!event.touches || event.touches.length < 2) {
            pinchState = null;
          }
        }, { passive: true });
      }

      var transparentInput = document.getElementById('molviewer-snapshot-transparent');
      var qualitySelect = document.getElementById('molviewer-snapshot-quality');
      var downloadButtons = [];
      var snapshotStatusNode = document.getElementById('molviewer-snapshot-status');

      function collectDownloadButtons() {
        var buttons = [];
        var mainButton = document.getElementById('molviewer-snapshot-download');
        var floatingButton = document.getElementById('molviewer-snapshot-download-float');
        if (mainButton) buttons.push(mainButton);
        if (floatingButton) buttons.push(floatingButton);
        return buttons;
      }

      function setDownloadButtonsDisabled(disabled) {
        downloadButtons.forEach(function(btn) { btn.disabled = !!disabled; });
      }

      function setSnapshotStatus(message, success) {
        if (!snapshotStatusNode) { snapshotStatusNode = document.getElementById('molviewer-snapshot-status'); }
        if (!snapshotStatusNode) return;
        snapshotStatusNode.textContent = message || '';
        if (!message) {
          snapshotStatusNode.style.color = cfg.theme.text;
          return;
        }
        snapshotStatusNode.style.color = success ? '#047857' : cfg.theme.highlight;
      }

      function triggerDownloadUrl(url, filename) {
        var link = document.createElement('a');
        link.href = url;
        link.download = filename || 'snapshot.png';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      }

      function handleSnapshotRequest(event) {
        if (event) { event.preventDefault(); event.stopPropagation(); }
        if (!stage) return;

        if (!transparentInput) transparentInput = document.getElementById('molviewer-snapshot-transparent');
        if (!qualitySelect) qualitySelect = document.getElementById('molviewer-snapshot-quality');
        if (!downloadButtons.length) downloadButtons = collectDownloadButtons();
        if (!downloadButtons.length) return;

        var transparent = transparentInput ? !!transparentInput.checked : false;
        var factorValue = qualitySelect ? parseFloat(qualitySelect.value || '1') : 1;
        if (!(factorValue > 0)) factorValue = 1;
        var antialias = cfg.snapshot && cfg.snapshot.antialias !== undefined ? !!cfg.snapshot.antialias : true;
        var trim = cfg.snapshot && cfg.snapshot.trim !== undefined ? !!cfg.snapshot.trim : true;
        var background = (!transparent && cfg.snapshot && cfg.snapshot.background) ? cfg.snapshot.background : cfg.theme.background;
        var filename = (cfg.snapshot && cfg.snapshot.filename) || 'snapshot.png';
        setSnapshotStatus('Rendering image...', false);
        setDownloadButtonsDisabled(true);

        stage.makeImage({ factor: factorValue, antialias: antialias, trim: trim, transparent: transparent, renderer: 'color', backgroundColor: transparent ? 'rgba(0,0,0,0)' : background }).then(function(image) {
          var completed = false;
          function finish(success, message) {
            if (completed) return;
            completed = true;
            setDownloadButtonsDisabled(false);
            if (success) {
              setSnapshotStatus('PNG downloaded', true);
            } else {
              setSnapshotStatus(message || 'Snapshot failed.', false);
            }
          }

          if (image instanceof Blob) {
            var blobUrl = URL.createObjectURL(image);
            triggerDownloadUrl(blobUrl, filename);
            setTimeout(function() { URL.revokeObjectURL(blobUrl); }, 4000);
            finish(true);
            return;
          }

          try {
            if (image && typeof image.download === 'function') {
              image.download(filename);
              finish(true);
              return;
            }
          } catch (err) {
            console.warn('Snapshot helper download failed', err);
          }

          var blobCandidate = image ? (typeof image.blob === 'function' ? image.blob() : image.blob) : null;
          if (blobCandidate instanceof Blob) {
            var url = URL.createObjectURL(blobCandidate);
            triggerDownloadUrl(url, filename);
            setTimeout(function() { URL.revokeObjectURL(url); }, 4000);
            finish(true);
            return;
          }

          var canvas = image ? (typeof image.getCanvas === 'function' ? image.getCanvas() : image.canvas) : null;
          if (canvas && canvas.toBlob) {
            canvas.toBlob(function(blob) {
              if (blob) {
                var url = URL.createObjectURL(blob);
                triggerDownloadUrl(url, filename);
                setTimeout(function() { URL.revokeObjectURL(url); }, 4000);
                finish(true);
              } else {
                finish(false, 'Snapshot unavailable.');
              }
            });
            return;
          }

          var dataUrl = image ? (typeof image.getDataURL === 'function' ? image.getDataURL() : image.dataURL) : null;
          if (typeof dataUrl === 'string') {
            triggerDownloadUrl(dataUrl, filename);
            finish(true);
            return;
          }

          finish(false, 'Snapshot unavailable.');
        }).catch(function(err) {
          console.error('Snapshot failed', err);
          setDownloadButtonsDisabled(false);
          setSnapshotStatus('Snapshot failed. Check console for details.', false);
        });
      }

      function setupSnapshotControls() {
        if (!transparentInput) transparentInput = document.getElementById('molviewer-snapshot-transparent');
        if (!qualitySelect) qualitySelect = document.getElementById('molviewer-snapshot-quality');
        var buttons = collectDownloadButtons();
        if (!buttons.length) return;
        downloadButtons = buttons;
        buttons.forEach(function(btn) {
          if (btn.hasAttribute('data-bound')) return;
          btn.setAttribute('data-bound', '1');
          btn.addEventListener('click', handleSnapshotRequest);
        });
      }

      function overlayContainer() {
        var node = document.getElementById('ngl-stage');
        if (!node) return null;
        var overlay = node.querySelector('.molviewer-overlay');
        if (overlay) return overlay;
        overlay = document.createElement('div');
        overlay.className = 'molviewer-overlay';
        Object.assign(overlay.style, {
          position: 'absolute', left: '10px', right: '10px', bottom: '10px', pointerEvents: 'none',
          background: 'rgba(15,23,42,0.78)', color: '#F8FAFC', fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
          fontSize: '12px', padding: '8px 12px', borderRadius: '8px', boxShadow: '0 8px 25px rgba(15,23,42,0.2)',
          maxHeight: '45%', overflowY: 'auto', lineHeight: '1.45'
        });
        node.appendChild(overlay);
        return overlay;
      }

      function setOverlay(lines) {
        var overlay = overlayContainer();
        if (!overlay) return;
        overlay.style.background = cfg.theme.background.toLowerCase() === '#0e1117' ? 'rgba(248,250,252,0.82)' : 'rgba(15,23,42,0.78)';
        overlay.style.color = cfg.theme.background.toLowerCase() === '#0e1117' ? '#0F172A' : '#F8FAFC';
        overlay.innerHTML = (lines || []).map(function(line) { return '<div>' + line + '</div>'; }).join('');
      }

      var blob = new Blob([cfg.pdb], { type: 'text/plain' });
      stage.loadFile(blob, { ext: 'pdb' }).then(function(component) {
        var styleName = cfg.style || 'ball+stick';
        var baseParams = Object.assign({}, cfg.styleParams || {});
        component.addRepresentation(styleName, baseParams);
        stage.autoView();

        var structure = component.structure;

        ensureZoomControls();
        enableGestureZoom();
        setupSnapshotControls();

        if (Array.isArray(cfg.labelModes) && cfg.labelModes.length) {
          var labelText = {};
          var hasLabels = false;
          cfg.atoms.forEach(function(atom) {
            var parts = [];
            if (cfg.labelModes.indexOf('symbol') !== -1) parts.push(atom.symbol);
            if (cfg.labelModes.indexOf('atomic_number') !== -1) parts.push('Z=' + atom.atomic_number);
            if (cfg.labelModes.indexOf('atom_index') !== -1) parts.push('#' + atom.index);
            if (parts.length) {
              labelText[atom.index] = parts.join(' · ');
              hasLabels = true;
            }
          });
          if (hasLabels) {
            var labelBackground = cfg.theme.background.toLowerCase() === '#0e1117' ? '#F8FAFC' : '#0F172A';
            component.addRepresentation('label', {
              labelType: 'text',
              labelText: labelText,
              labelGrouping: 'atom',
              attachment: 'middle-center',
              color: cfg.theme.text,
              showBackground: true,
              backgroundColor: labelBackground,
              backgroundOpacity: 0.35,
              fixedSize: true,
              zOffset: 2
            });
          }
        }

        function refreshHighlights() {
          clearReprs(highlightReprs);
          selection.forEach(function(sel, idx) {
            var repr = component.addRepresentation('ball+stick', {
              sele: '@' + sel.index,
              color: cfg.palette[idx % cfg.palette.length],
              radiusType: 'size',
              radiusSize: cfg.bondRadius,
              aspectRatio: cfg.highlightAspectRatio
            });
            highlightReprs.push(repr);
          });
        }

        function addDistancePairs(pairs, color) {
          if (!pairs.length) return;
          var repr = component.addRepresentation('distance', {
            atomPair: pairs.map(function(pair) { return pair.map(function(idx) { return structure.getAtomProxy(idx); }); }),
            labelVisible: true,
            color: color || cfg.palette[0]
          });
          measurementReprs.push(repr);
        }

        function measurementKind() {
          if (cfg.mode !== 'measurement') return null;
          var count = selection.length;
          if (count >= 4) return 'dihedral';
          if (count === 3) return 'angle';
          if (count === 2) return 'distance';
          return null;
        }

        function refreshMeasurements() {
          clearReprs(measurementReprs);
          if (cfg.mode !== 'measurement') return;
          var kind = measurementKind();
          if (kind === 'distance') {
            var ids = selection.slice(-2).map(function(sel) { return sel.index; });
            addDistancePairs([[ids[0], ids[1]]], cfg.palette[0]);
          } else if (kind === 'angle') {
            var ida = selection.slice(-3).map(function(sel) { return sel.index; });
            addDistancePairs([[ida[0], ida[1]], [ida[1], ida[2]]], cfg.palette[0]);
          } else if (kind === 'dihedral') {
            var q = selection.slice(-4);
            var subtract = function(a,b) { return [a.x-b.x, a.y-b.y, a.z-b.z]; };
            var cross = function(a,b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; };
            var norm = function(v) { var n=Math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); return n? [v[0]/n,v[1]/n,v[2]/n]:[0,0,0]; };
            var b0 = subtract(q[1], q[0]);
            var b1 = subtract(q[2], q[1]);
            var b2 = subtract(q[3], q[2]);
            var n1 = norm(cross(b0, b1));
            var n2 = norm(cross(b1, b2));
            var m1 = cross(n1, norm(b1));
            var x = n1[0]*n2[0]+n1[1]*n2[1]+n1[2]*n2[2];
            var y = m1[0]*n2[0]+m1[1]*n2[1]+m1[2]*n2[2];
            var dih = Math.atan2(y, x) * 180/Math.PI;
            var lines = ['Dihedral = ' + dih.toFixed(2) + ' deg'];
          }
        }

        function clearSelection() {
          selection.length = 0;
          refreshHighlights();
          refreshMeasurements();
          summaryLines();
        }

        function summaryLines() {
          var lines = [];
          lines.push('<strong>Mode:</strong> ' + cfg.modeLabel);
          if (cfg.mode === 'select' && selection.length) {
            var atom = selection[selection.length-1];
            lines.push('[' + atom.serial + '] ' + atom.symbol + ' (index ' + atom.index + ', Z=' + atom.atomic_number + ')');
            lines.push('Coordinates: (' + atom.x.toFixed(3) + ', ' + atom.y.toFixed(3) + ', ' + atom.z.toFixed(3) + ') Angstrom');
            lines.push('Mass: ' + atom.mass.toFixed(3) + ' amu');
          } else if (cfg.mode === 'measurement') {
            var kind = measurementKind();
            if (kind === 'distance') {
              var a = selection[selection.length-2];
              var b = selection[selection.length-1];
              var dist = Math.sqrt(Math.pow(a.x-b.x,2)+Math.pow(a.y-b.y,2)+Math.pow(a.z-b.z,2));
              lines.push('Distance = ' + dist.toFixed(3) + ' Angstrom');
            } else if (kind === 'angle') {
              var p = selection.slice(-3);
              var v1 = [p[0].x-p[1].x, p[0].y-p[1].y, p[0].z-p[1].z];
              var v2 = [p[2].x-p[1].x, p[2].y-p[1].y, p[2].z-p[1].z];
              var dot = v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2];
              var n1 = Math.sqrt(v1[0]*v1[0]+v1[1]*v1[1]+v1[2]*v1[2]);
              var n2 = Math.sqrt(v2[0]*v2[0]+v2[1]*v2[1]+v2[2]*v2[2]);
              var ang = Math.acos(Math.max(-1, Math.min(1, dot/(n1*n2)))) * 180/Math.PI;
              lines.push('Angle = ' + ang.toFixed(2) + ' deg');
            } else if (kind === 'dihedral') {
              var q = selection.slice(-4);
              var subtract = function(a,b) { return [a.x-b.x, a.y-b.y, a.z-b.z]; };
              var cross = function(a,b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; };
              var norm = function(v) { var n=Math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); return n? [v[0]/n,v[1]/n,v[2]/n]:[0,0,0]; };
              var b0 = subtract(q[1], q[0]);
              var b1 = subtract(q[2], q[1]);
              var b2 = subtract(q[3], q[2]);
              var n1 = norm(cross(b0, b1));
              var n2 = norm(cross(b1, b2));
              var m1 = cross(n1, norm(b1));
              var x = n1[0]*n2[0]+n1[1]*n2[1]+n1[2]*n2[2];
              var y = m1[0]*n2[0]+m1[1]*n2[1]+m1[2]*n2[2];
              var dih = Math.atan2(y, x) * 180/Math.PI;
              lines.push('Dihedral = ' + dih.toFixed(2) + ' deg');
            }
          }

          if (cfg.mode !== 'rotate' && cfg.maxAtoms > 0 && selection.length < cfg.maxAtoms) {
            var remaining = cfg.maxAtoms - selection.length;
            lines.push('Select ' + remaining + ' more atom' + (remaining > 1 ? 's' : '') + ' ...');
          }
          setOverlay(lines);
        }

        function toggleSelection(atom) {
          var idx = selection.findIndex(function(sel) { return sel.index === atom.index; });
          if (idx >= 0) selection.splice(idx, 1);
          else {
            if (cfg.maxAtoms > 0 && selection.length >= cfg.maxAtoms) selection.shift();
            selection.push(atom);
          }
          refreshHighlights();
          refreshMeasurements();
          summaryLines();
        }

        var viewerContainer = stage.viewer && stage.viewer.container ? stage.viewer.container : null;
        if (viewerContainer && !viewerContainer.dataset.molviewerMeasurementResetBound) {
          viewerContainer.dataset.molviewerMeasurementResetBound = '1';
          var measurementResetHandler = function(event) {
            if (cfg.mode !== 'measurement') return;
            if (event) { event.preventDefault(); event.stopPropagation(); }
            clearSelection();
          };
          viewerContainer.addEventListener('contextmenu', measurementResetHandler);
          viewerContainer.addEventListener('mousedown', function(event) {
            if (!event || event.button !== 2) return;
            measurementResetHandler(event);
          });
        }

        if (cfg.mode !== 'rotate' && cfg.maxAtoms > 0) {
          stage.signals.clicked.add(function(pickingProxy) {
            if (!pickingProxy || !pickingProxy.atom) return;
            var atom = cfg.atoms[pickingProxy.atom.index];
            if (!atom) return;
            toggleSelection(atom);
          });
        }

        refreshHighlights();
        refreshMeasurements();
        summaryLines();
      }).catch(function(err) {
        console.error('NGL load failed', err);
        setOverlay(['Failed to load structure: ' + err]);
      });
    })();
    """

    # Inject payload safely without f-string brace parsing
    js = js.replace("__PAYLOAD__", json.dumps(payload))

    # Now build the outer HTML with small f-strings (only where needed)
    return (
        '<div style="display:flex;justify-content:center;">'
        f'  <div id="molviewer-stage-wrapper" style="display:flex;flex-direction:column;align-items:stretch;width:{width}px;gap:12px;">'
        f'    <div id="ngl-stage" style="width:{width}px;height:{height}px;position:relative;background:{theme.background};border-radius:12px;overflow:hidden;">'
        f'      <button id="molviewer-snapshot-download-float" type="button" title="Download PNG" style="position:absolute;top:12px;left:12px;padding:6px 12px;border-radius:6px;border:0;background:{theme.highlight};color:#FFFFFF;font-weight:600;cursor:pointer;box-shadow:0 4px 12px rgba(15,23,42,0.18);z-index:25;pointer-events:auto;">Save PNG</button>'
        f'    </div>'
        f'    <div id="molviewer-snapshot-bar" style="display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:8px;padding:8px 12px;border:1px solid rgba(15,23,42,0.12);border-radius:10px;background:{theme.plot_bg};box-shadow:0 4px 16px rgba(15,23,42,0.12);">'
        f'      <div style="display:flex;flex-wrap:wrap;align-items:center;gap:12px;">'
        f'        <label style="display:flex;align-items:center;gap:6px;font-size:13px;cursor:pointer;">'
        f'          <input id="molviewer-snapshot-transparent" type="checkbox" style="margin:0;">'
        f'          Transparent background'
        f'        </label>'
        f'        <label style="display:flex;align-items:center;gap:6px;font-size:13px;">'
        f'          Quality'
        f'          <select id="molviewer-snapshot-quality" style="padding:4px 8px;border-radius:6px;border:1px solid rgba(15,23,42,0.18);background:#FFFFFF;min-width:140px;">'
        f'            {quality_options_html}'
        f'          </select>'
        f'        </label>'
        f'      </div>'
        f'      <div style="display:flex;align-items:center;gap:10px;">'
        f'        <button id="molviewer-snapshot-download" type="button" style="padding:8px 16px;border-radius:6px;border:0;background:{theme.highlight};color:#FFFFFF;font-weight:600;cursor:pointer;box-shadow:0 4px 12px rgba(15,23,42,0.18);">Download PNG</button>'
        f'      </div>'
        f'    </div>'
        f'    <div id="molviewer-snapshot-status" style="font-size:12px;color:{theme.text_color};min-height:14px;"></div>'
        f'  </div>'
        f'</div>'
        '<script src="https://unpkg.com/ngl@2.0.0-dev.39/dist/ngl.js"></script>'
        '<script>' + js + '</script>'
    )

import numpy as np

def atoms_to_xyz_block(atoms: Atoms) -> str:
    buffer = StringIO()
    ase_write(buffer, atoms, format="xyz")
    return buffer.getvalue()

def theme_transparent(color: str) -> str:
    """Return a transparent RGBA hex string based on a solid hex color."""

    if color.startswith("#") and len(color) == 7:
        return f"{color}66"
    return "#00000000"

def add_atom_labels(
    viewer: Any,
    atoms: Atoms,
    label_modes: List[str],
    theme: ThemeConfig,
) -> None:
    """Annotate atoms with labels based on the selected display modes."""

    modes = {mode.replace(" ", "_").lower() for mode in label_modes}
    if not modes:
        return

    symbols = atoms.get_chemical_symbols()
    numbers = atoms.get_atomic_numbers()
    coords = atoms.get_positions()

    for idx, (symbol, atomic_number) in enumerate(zip(symbols, numbers)):
        parts: list[str] = []
        if "symbol" in modes:
            parts.append(symbol)
        if "atomic_number" in modes:
            parts.append(f"Z={int(atomic_number)}")
        if "atom_index" in modes:
            parts.append(f"i={idx}")
        if not parts:
            continue

        x, y, z = map(float, coords[idx])
        viewer.addLabel(
            " | ".join(parts),
            {
                "position": {"x": x, "y": y, "z": z},
                "fontSize": 14,
                "fontColor": theme.highlight,
                "backgroundColor": theme_transparent(theme.background),
                "backgroundOpacity": 0.6,
                "borderThickness": 0,
                "alignment": "center",
                "inFront": True,
            },
        )

def add_axes(viewer: Any, atoms: Atoms, scale: float) -> None:
    """Add RGB axes arrows to the viewer based on the molecule span."""

    positions = atoms.get_positions()
    span = float(np.ptp(positions, axis=0).max()) or 1.0
    axis_length = span * scale

    min_coords = positions.min(axis=0)
    offset = 0.1 * span
    origin = {
        "x": float(min_coords[0] - offset),
        "y": float(min_coords[1] - offset),
        "z": float(min_coords[2] - offset),
    }
    arrows = [
        ("X-axis", {"x": axis_length, "y": 0.0, "z": 0.0}, "#FF4136"),
        ("Y-axis", {"x": 0.0, "y": axis_length, "z": 0.0}, "#2ECC40"),
        ("Z-axis", {"x": 0.0, "y": 0.0, "z": axis_length}, "#0074D9"),
    ]
    for name, end, color in arrows:
        arrow_end = {
            "x": origin["x"] + end["x"],
            "y": origin["y"] + end["y"],
            "z": origin["z"] + end["z"],
        }
        viewer.addArrow(
            {
                "start": origin,
                "end": arrow_end,
                "color": color,
                "radius": span * 0.008,
            }
        )
        viewer.addLabel(
            name,
            {
                "fontColor": color,
                "backgroundColor": theme_transparent(color),
                "fontSize": 14,
                "position": arrow_end,
                "inFront": True,
            },
        )

def render_3dmol_view(
    atoms: Atoms,
    label: str,
    *,
    theme: ThemeConfig,
    height: int,
    width: int = 700,
    threedmol_style: str = "stick",
    threedmol_atom_radius: Optional[float] = None,
    threedmol_bond_radius: Optional[float] = None,
    show_axes: bool = False,
    axis_scale: float = 0.4,
    label_modes: Optional[List[str]] = None,
    container_id: Optional[str] = None,
) -> str:
    xyz_block = atoms_to_xyz_block(atoms)

    payload = {
        "xyz": xyz_block,
        "style": threedmol_style,
        "width": width,
        "height": height,
        "label": label,
        "background": theme.background,
        "atom_radius": threedmol_atom_radius,
        "bond_radius": threedmol_bond_radius,
        "show_axes": show_axes,
        "axis_scale": axis_scale,
        "label_modes": label_modes,
        "theme_highlight": theme.highlight,
        "theme_background_transparent": theme_transparent(theme.background),
    }

    js = """
    function init3Dmol() {
        var cfg = __PAYLOAD__;
        let viewer = $3Dmol.createViewer('threedmol-viewer', { backgroundColor: cfg.background });
        viewer.addModel(cfg.xyz, "xyz");
        if (cfg.style === "Ball and Stick") {
            viewer.setStyle({}, {stick: {radius: cfg.bond_radius}, sphere: {radius: cfg.atom_radius}});
        } else {
            viewer.setStyle({}, { [cfg.style]: {} });
        }
        viewer.zoomTo();

        if (cfg.show_axes) {
            // Add axes - simplified for 3Dmol, as it doesn't have a direct addArrow
            // This would require more complex JS to draw lines/cones
            // For now, we'll just add labels for the axes
            var positions = viewer.getModel().selectedAtoms({});
            var min_x = Infinity, min_y = Infinity, min_z = Infinity;
            var max_x = -Infinity, max_y = -Infinity, max_z = -Infinity;
            for (var i = 0; i < positions.length; i++) {
                var p = positions[i].coord;
                min_x = Math.min(min_x, p.x); max_x = Math.max(max_x, p.x);
                min_y = Math.min(min_y, p.y); max_y = Math.max(max_y, p.y);
                min_z = Math.min(min_z, p.z); max_z = Math.max(max_z, p.z);
            }
            var span = Math.max(max_x - min_x, max_y - min_y, max_z - min_z) || 1.0;
            var axis_length = span * cfg.axis_scale;
            var offset = 0.1 * span;
            var origin = {x: min_x - offset, y: min_y - offset, z: min_z - offset};

            var arrows = [
                {name: "X-axis", end: {x: axis_length, y: 0.0, z: 0.0}, color: "#FF4136"},
                {name: "Y-axis", end: {x: 0.0, y: axis_length, z: 0.0}, color: "#2ECC40"},
                {name: "Z-axis", end: {x: 0.0, y: 0.0, z: axis_length}, color: "#0074D9"},
            ];

            arrows.forEach(function(arrow) {
                var arrow_end = {
                    x: origin.x + arrow.end.x,
                    y: origin.y + arrow.end.y,
                    z: origin.z + arrow.end.z
                };
                // 3Dmol doesn't have a direct arrow function, so we'll use labels for now
                viewer.addLabel(
                    arrow.name,
                    {
                        position: arrow_end,
                        fontColor: arrow.color,
                        backgroundColor: cfg.theme_background_transparent,
                        fontSize: 14,
                        alignment: "center",
                        inFront: true,
                    }
                );
            });
        }

        if (cfg.label_modes && cfg.label_modes.length > 0) {
            // Atomic number lookup table
            const atomicNumbers = {
                "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
                "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
                "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26,
                "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34,
                "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42,
                "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
                "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58,
                "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66,
                "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72, "Ta": 73, "W": 74,
                "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82,
                "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90,
                "Pa": 91, "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98,
                "Es": 99, "Fm": 100, "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106,
                "Bh": 107, "Hs": 108, "Mt": 109, "Ds": 110, "Rg": 111, "Cn": 112, "Nh": 113, "Fl": 114,
                "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118
            };

            var atoms = viewer.getModel().selectedAtoms({});
            atoms.forEach(function(atom) {
                var parts = [];
                var symbol = atom.elem;
                var atomic_number = atomicNumbers[symbol] || "undefined"; // Get atomic number from lookup
                var index = atom.serial - 1; // Use atom.serial (1-based) for 0-based index

                var modes = cfg.label_modes.map(m => m.toLowerCase().replace(" ", "_"));

                if (modes.includes("symbol")) {
                    parts.push(symbol);
                }
                if (modes.includes("atomic_number")) {
                    parts.push("Z=" + atomic_number);
                }
                if (modes.includes("atom_index")) {
                    parts.push("i=" + index);
                }

                if (parts.length > 0) {
                    viewer.addLabel(
                        parts.join(" | "),
                        {
                            position: atom,
                            fontSize: 14,
                            fontColor: cfg.theme_highlight,
                            backgroundColor: cfg.theme_background_transparent,
                            backgroundOpacity: 0.6,
                            borderThickness: 0,
                            alignment: "center",
                            inFront: true,
                        }
                    );
                }
            });
        }

        viewer.render();
    }

    function load3Dmol() {
        if (typeof $3Dmol !== 'undefined') {
            init3Dmol();
            return;
        }
        var script = document.createElement('script');
        script.src = "https://cdnjs.cloudflare.com/ajax/libs/3Dmol/1.4.0/3Dmol-min.js";
        script.onload = init3Dmol;
        document.head.appendChild(script);
    }

    load3Dmol();
    """
    js = js.replace("__PAYLOAD__", json.dumps(payload))

    dom_id = _safe_dom_id(container_id, default="threedmol-viewer")
    js = js.replace("threedmol-viewer", dom_id)

    return (
        f'<div id="{dom_id}" style="width: {width}px; height: {height}px; position: relative; margin: 0 auto;"></div>'
        f'<script>{js}</script>'
    )

