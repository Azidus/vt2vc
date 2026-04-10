print("VT2VC")

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import time

print("V1.1_20260409")


# ============================================================
# CONFIG
# ============================================================
OBJ_PATH = ""

# Used only if a material has neither usable map_Kd nor Kd
FALLBACK_RGB = (200, 200, 200)

# If True, triangulate non-triangle faces before output
TRIANGULATE = True

# Output keeps vt and writes faces as v/vt
# Vertex colors are written on v lines: v x y z r g b
WRITE_VT = True


# ============================================================
# DATA STRUCTURES
# ============================================================
@dataclass(frozen=True)
class FaceVertex:
    v: Optional[int]
    vt: Optional[int]
    vn: Optional[int]
    original_token: str


@dataclass
class Face:
    verts: List[FaceVertex]
    material_name: Optional[str]
    object_name: Optional[str]
    group_name: Optional[str]
    smoothing: Optional[str]


# ============================================================
# PATH RESOLUTION
# ============================================================
def resolve_obj_path() -> Path:
    if OBJ_PATH.strip():
        p = Path(OBJ_PATH)
        if not p.exists():
            raise FileNotFoundError(f"OBJ_PATH does not exist: {p}")
        return p

    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if not p.exists():
            raise FileNotFoundError(f"Command line OBJ does not exist: {p}")
        return p

    obj_files = sorted(list(Path.cwd().glob("*.obj")) + list(Path.cwd().glob("*.OBJ")))
    if not obj_files:
        raise FileNotFoundError("No .obj/.OBJ file found in current working directory.")

    print(f"No OBJ_PATH or CLI arg given. Using: {obj_files[0]}")
    return obj_files[0]


def find_case_insensitive_file(parent: Path, name: str) -> Optional[Path]:
    """
    Check only this exact folder/path, case-insensitively where practical.
    If `name` includes subfolders, try that exact relative path first.
    """
    candidate = parent / name
    if candidate.exists() and candidate.is_file():
        print(f"Found at specified path: {candidate}")
        return candidate

    name_path = Path(name)

    if len(name_path.parts) == 1:
        target = name_path.name.lower()
        for f in parent.iterdir():
            if f.is_file() and f.name.lower() == target:
                print(f"Found in exact folder (case-insensitive): {f}")
                return f

    return None


def find_file_near_obj(obj_folder: Path, name: str) -> Optional[Path]:
    """
    Search order:
    1. exact path as specified in the OBJ/MTL
    2. same folder as OBJ, by filename only
    3. any subfolder under OBJ folder, by filename only
    """
    name = name.strip().replace("\\", "/")
    if not name:
        return None

    print(f"Trying specified path: {obj_folder / name}")
    found = find_case_insensitive_file(obj_folder, name)
    if found is not None:
        return found

    target_name = Path(name).name
    target_name_lower = target_name.lower()

    print(f"Not found at specified path. Trying OBJ folder: {obj_folder / target_name}")
    for f in obj_folder.iterdir():
        if f.is_file() and f.name.lower() == target_name_lower:
            print(f"Found in OBJ folder: {f}")
            return f

    print(f"Not found in OBJ folder. Searching subfolders for: {target_name}")
    for f in obj_folder.rglob("*"):
        if f.is_file() and f.name.lower() == target_name_lower:
            print(f"Found in subfolder: {f}")
            return f

    print(f"File not found anywhere: {name}")
    return None


# ============================================================
# OBJ PARSING
# ============================================================
def parse_face_token(token: str) -> FaceVertex:
    parts = token.split("/")

    v = int(parts[0]) if len(parts) >= 1 and parts[0] else None
    vt = int(parts[1]) if len(parts) >= 2 and parts[1] else None
    vn = int(parts[2]) if len(parts) >= 3 and parts[2] else None

    return FaceVertex(v=v, vt=vt, vn=vn, original_token=token)


def parse_obj(obj_path: Path):
    """
    Keeps:
    - header lines before first face, excluding old mtllib/usemtl
    - all v/vt data
    - faces with active material/object/group/smoothing state
    - referenced mtllib filenames
    """
    header_lines: List[str] = []
    vertices: List[Tuple[float, float, float]] = []
    texcoords: List[Tuple[float, float]] = []
    faces: List[Face] = []
    mtllibs: List[str] = []

    seen_first_face = False

    current_material: Optional[str] = None
    current_object: Optional[str] = None
    current_group: Optional[str] = None
    current_smoothing: Optional[str] = None

    with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            stripped = line.strip()

            if not stripped:
                if not seen_first_face:
                    header_lines.append(line)
                continue

            parts = stripped.split()
            head = parts[0]

            if head == "mtllib":
                mtllibs.extend(parts[1:])

            elif head == "usemtl":
                current_material = " ".join(parts[1:]) if len(parts) > 1 else None

            elif head == "v":
                if len(parts) < 4:
                    raise ValueError(f"Invalid v line: {line}")
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
                if not seen_first_face:
                    header_lines.append(line)

            elif head == "vt":
                if len(parts) < 3:
                    raise ValueError(f"Invalid vt line: {line}")
                texcoords.append((float(parts[1]), float(parts[2])))
                if not seen_first_face:
                    header_lines.append(line)

            elif head == "o":
                current_object = " ".join(parts[1:]) if len(parts) > 1 else None
                if not seen_first_face:
                    header_lines.append(line)

            elif head == "g":
                current_group = " ".join(parts[1:]) if len(parts) > 1 else None
                if not seen_first_face:
                    header_lines.append(line)

            elif head == "s":
                current_smoothing = parts[1] if len(parts) > 1 else None
                if not seen_first_face:
                    header_lines.append(line)

            elif head == "f":
                seen_first_face = True
                face_verts = [parse_face_token(tok) for tok in parts[1:]]
                faces.append(
                    Face(
                        verts=face_verts,
                        material_name=current_material,
                        object_name=current_object,
                        group_name=current_group,
                        smoothing=current_smoothing,
                    )
                )

            else:
                if not seen_first_face:
                    header_lines.append(line)

    return header_lines, vertices, texcoords, faces, mtllibs


# ============================================================
# TRIANGULATION
# ============================================================
def triangulate_face(face: Face) -> List[Face]:
    if len(face.verts) < 3:
        raise ValueError("Face has fewer than 3 vertices.")

    if len(face.verts) == 3:
        return [face]

    tris: List[Face] = []
    for i in range(1, len(face.verts) - 1):
        tris.append(
            Face(
                verts=[face.verts[0], face.verts[i], face.verts[i + 1]],
                material_name=face.material_name,
                object_name=face.object_name,
                group_name=face.group_name,
                smoothing=face.smoothing,
            )
        )
    return tris


def triangulate_faces(faces: List[Face]) -> Tuple[List[Face], int]:
    triangulated: List[Face] = []
    changed_count = 0

    for face in faces:
        if len(face.verts) != 3:
            changed_count += 1
        triangulated.extend(triangulate_face(face))

    return triangulated, changed_count


# ============================================================
# MTL PARSING
# ============================================================
def resolve_mtl_paths(obj_path: Path, mtllibs: List[str]) -> List[Path]:
    found: List[Path] = []
    seen = set()

    for name in mtllibs:
        p = find_file_near_obj(obj_path.parent, name)
        if p is not None and p not in seen:
            found.append(p)
            seen.add(p)

    if not found:
        for candidate_name in [obj_path.with_suffix(".mtl").name, obj_path.with_suffix(".MTL").name]:
            p = find_file_near_obj(obj_path.parent, candidate_name)
            if p is not None and p not in seen:
                found.append(p)
                seen.add(p)

    return found


def parse_mtl_files(mtl_paths: List[Path]) -> Dict[str, Dict[str, object]]:
    """
    Returns:
      {
        material_name: {
          "map_Kd": str | None,
          "Kd": (r,g,b) | None
        }
      }
    """
    materials: Dict[str, Dict[str, object]] = {}

    for mtl_path in mtl_paths:
        current_material: Optional[str] = None

        with mtl_path.open("r", encoding="utf-8", errors="ignore") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                head = parts[0]

                if head == "newmtl":
                    current_material = " ".join(parts[1:]) if len(parts) > 1 else None
                    if current_material:
                        materials.setdefault(current_material, {"map_Kd": None, "Kd": None})

                elif head == "map_Kd" and current_material:
                    tex_name = line[len("map_Kd"):].strip()
                    if tex_name:
                        materials[current_material]["map_Kd"] = tex_name

                elif head == "Kd" and current_material:
                    if len(parts) >= 4:
                        r = max(0, min(255, int(round(float(parts[1]) * 255))))
                        g = max(0, min(255, int(round(float(parts[2]) * 255))))
                        b = max(0, min(255, int(round(float(parts[3]) * 255))))
                        materials[current_material]["Kd"] = (r, g, b)

    return materials


def resolve_material_textures(
    obj_path: Path,
    mtl_paths: List[Path],
    material_defs: Dict[str, Dict[str, object]],
) -> Dict[str, Optional[Path]]:
    material_to_texture: Dict[str, Optional[Path]] = {}

    for material_name, data in material_defs.items():
        tex_name = data.get("map_Kd")
        if not tex_name:
            material_to_texture[material_name] = None
            continue

        resolved = None

        # First try relative to each MTL file's folder
        for mtl_path in mtl_paths:
            resolved = find_file_near_obj(mtl_path.parent, str(tex_name))
            if resolved is not None:
                break

        # Then try relative to OBJ folder
        if resolved is None:
            resolved = find_file_near_obj(obj_path.parent, str(tex_name))

        material_to_texture[material_name] = resolved

    return material_to_texture


# ============================================================
# UV / TEXTURE SAMPLING
# ============================================================
def uv_to_pixel(uv: Tuple[float, float], width: int, height: int) -> Tuple[int, int]:
    u, v = uv
    u = u % 1.0
    v = v % 1.0

    x = int(round(u * (width - 1)))
    y = int(round((1.0 - v) * (height - 1)))

    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    return x, y


def sample_texture_at_uv(image_rgb: np.ndarray, uv: Tuple[float, float]) -> Tuple[int, int, int]:
    h, w, _ = image_rgb.shape
    x, y = uv_to_pixel(uv, w, h)
    rgb = image_rgb[y, x, :3]
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


# ============================================================
# REPORTING
# ============================================================
def print_model_report(original_faces: List[Face], output_faces: List[Face], changed_count: int) -> None:
    materials_used = sorted({f.material_name for f in original_faces if f.material_name})
    object_count = len({f.object_name for f in original_faces if f.object_name is not None})
    group_count = len({f.group_name for f in original_faces if f.group_name is not None})
    smoothing_count = len({f.smoothing for f in original_faces if f.smoothing is not None})

    print()
    print("========== MODEL REPORT ==========")
    print(f"Original faces on OBJ model: {len(original_faces)}")
    print(f"Faces triangulated: {changed_count}")
    print(f"Faces after triangulation: {len(output_faces)}")
    print(f"Materials referenced by faces: {len(materials_used)}")
    print(f"Objects found: {object_count}")
    print(f"Groups found: {group_count}")
    print(f"Smoothing states found: {smoothing_count}")
    print("==================================")
    print()


def print_texture_reports(texture_cache: Dict[Path, np.ndarray]) -> None:
    print("========== TEXTURE REPORT ==========")
    if not texture_cache:
        print("No textures loaded.")
    else:
        for tex_path, image_rgb in texture_cache.items():
            h, w, _ = image_rgb.shape
            total_pixels = w * h
            unique_color_count = len(np.unique(image_rgb.reshape(-1, 3), axis=0))
            print(f"Texture: {tex_path.name}")
            print(f"  Pixels: {w} * {h} = {total_pixels}")
            print(f"  Unique colors: {unique_color_count}")
    print("====================================")
    print()


# ============================================================
# VERTEX COLOR BUILD
# ============================================================
def build_vertex_colored_mesh(
    source_vertices: List[Tuple[float, float, float]],
    source_texcoords: List[Tuple[float, float]],
    faces: List[Face],
    material_defs: Dict[str, Dict[str, object]],
    material_to_texture: Dict[str, Optional[Path]],
    texture_cache: Dict[Path, np.ndarray],
) -> Tuple[
    List[Tuple[float, float, float, float, float, float]],
    List[Tuple[float, float]],
    List[Face],
    int,
    int,
    int,
]:
    """
    Builds a new mesh where every output vertex can have its own RGB color.

    Because OBJ vertex color is attached to `v`, not `vt`, we duplicate vertices
    when needed so different UV/color corners can coexist safely.
    """
    out_vertices: List[Tuple[float, float, float, float, float, float]] = []
    out_texcoords: List[Tuple[float, float]] = []
    out_faces: List[Face] = []

    # dedupe by full corner identity after color is known
    # key = (source_v, source_vt, rgb)
    corner_to_new_indices: Dict[Tuple[int, Optional[int], Tuple[int, int, int]], Tuple[int, Optional[int]]] = {}

    texture_count = 0
    kd_count = 0
    fallback_count = 0

    for face_i, face in enumerate(faces, start=1):
        new_face_verts: List[FaceVertex] = []

        tex_path = None
        kd = None

        if face.material_name is not None and face.material_name in material_defs:
            tex_path = material_to_texture.get(face.material_name)
            kd = material_defs[face.material_name].get("Kd")

        for corner in face.verts:
            if corner.v is None:
                raise ValueError("Face corner missing vertex index.")

            color: Tuple[int, int, int]

            if tex_path is not None:
                if corner.vt is None:
                    color = FALLBACK_RGB
                    fallback_count += 1
                else:
                    uv = source_texcoords[corner.vt - 1]
                    color = sample_texture_at_uv(texture_cache[tex_path], uv)
                    texture_count += 1
            elif kd is not None:
                color = kd  # type: ignore[assignment]
                kd_count += 1
            else:
                color = FALLBACK_RGB
                fallback_count += 1

            key = (corner.v, corner.vt, color)

            if key in corner_to_new_indices:
                new_v_idx, new_vt_idx = corner_to_new_indices[key]
            else:
                x, y, z = source_vertices[corner.v - 1]
                r, g, b = color

                out_vertices.append((x, y, z, r / 255.0, g / 255.0, b / 255.0))
                new_v_idx = len(out_vertices)

                new_vt_idx = None
                if WRITE_VT and corner.vt is not None:
                    u, v = source_texcoords[corner.vt - 1]
                    out_texcoords.append((u, v))
                    new_vt_idx = len(out_texcoords)

                corner_to_new_indices[key] = (new_v_idx, new_vt_idx)

            new_face_verts.append(
                FaceVertex(
                    v=new_v_idx,
                    vt=new_vt_idx,
                    vn=None,
                    original_token="",
                )
            )

        out_faces.append(
            Face(
                verts=new_face_verts,
                material_name=face.material_name,
                object_name=face.object_name,
                group_name=face.group_name,
                smoothing=face.smoothing,
            )
        )

        print(f"Face {face_i}/{len(faces)} processed.")

    return out_vertices, out_texcoords, out_faces, texture_count, kd_count, fallback_count


# ============================================================
# OUTPUT WRITING
# ============================================================
def write_vertex_color_obj(
    out_obj_path: Path,
    header_lines: List[str],
    out_vertices: List[Tuple[float, float, float, float, float, float]],
    out_texcoords: List[Tuple[float, float]],
    out_faces: List[Face],
) -> None:
    with out_obj_path.open("w", encoding="utf-8") as f:
        # Keep non-geometry header lines only
        for line in header_lines:
            stripped = line.strip()
            if (
                stripped.startswith("v ")
                or stripped.startswith("vt ")
                or stripped.startswith("vn ")
                or stripped.startswith("usemtl ")
            ):
                continue
            f.write(line + "\n")

        # New vertex list with RGB appended
        for x, y, z, r, g, b in out_vertices:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f} {r:.8f} {g:.8f} {b:.8f}\n")

        if WRITE_VT:
            for u, v in out_texcoords:
                f.write(f"vt {u:.6f} {v:.6f}\n")

        last_object: Optional[str] = None
        last_group: Optional[str] = None
        last_smoothing: Optional[str] = None
        last_material: Optional[str] = None

        for face in out_faces:
            if face.object_name != last_object:
                if face.object_name is not None:
                    f.write(f"o {face.object_name}\n")
                last_object = face.object_name

            if face.group_name != last_group:
                if face.group_name is not None:
                    f.write(f"g {face.group_name}\n")
                last_group = face.group_name

            if face.smoothing != last_smoothing:
                if face.smoothing is not None:
                    f.write(f"s {face.smoothing}\n")
                last_smoothing = face.smoothing

            if face.material_name != last_material:
                if face.material_name is not None:
                    f.write(f"usemtl {face.material_name}\n")
                last_material = face.material_name

            tokens: List[str] = []
            for fv in face.verts:
                if WRITE_VT and fv.vt is not None:
                    tokens.append(f"{fv.v}/{fv.vt}")
                else:
                    tokens.append(str(fv.v))

            f.write("f " + " ".join(tokens) + "\n")


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    obj_path = resolve_obj_path()
    header_lines, source_vertices, source_texcoords, original_faces, mtllibs = parse_obj(obj_path)

    if not original_faces:
        raise ValueError("No faces found in OBJ.")

    output_faces = original_faces
    triangulated_count = 0
    if TRIANGULATE:
        output_faces, triangulated_count = triangulate_faces(original_faces)

    print(f"OBJ: {obj_path}")
    print_model_report(original_faces, output_faces, triangulated_count)

    mtl_paths = resolve_mtl_paths(obj_path, mtllibs)
    if mtl_paths:
        print("MTL files found:")
        for p in mtl_paths:
            print(f"  {p}")
    else:
        print("No MTL files found. Faces will use fallback color.")

    material_defs = parse_mtl_files(mtl_paths) if mtl_paths else {}
    material_to_texture = resolve_material_textures(obj_path, mtl_paths, material_defs) if mtl_paths else {}

    print()
    print("========== MATERIAL -> SOURCE ==========")
    if material_defs:
        for mat_name in sorted(material_defs):
            tex_path = material_to_texture.get(mat_name)
            kd = material_defs[mat_name].get("Kd")
            if tex_path is not None:
                print(f"{mat_name} -> texture: {tex_path.name}")
            elif kd is not None:
                print(f"{mat_name} -> Kd: {kd}")
            else:
                print(f"{mat_name} -> fallback")
    else:
        print("No materials parsed.")
    print("========================================")
    print()

    texture_cache: Dict[Path, np.ndarray] = {}
    for tex_path in sorted({p for p in material_to_texture.values() if p is not None}):
        texture_cache[tex_path] = np.array(Image.open(tex_path).convert("RGB"))

    print_texture_reports(texture_cache)

    out_vertices, out_texcoords, out_faces, texture_count, kd_count, fallback_count = build_vertex_colored_mesh(
        source_vertices=source_vertices,
        source_texcoords=source_texcoords,
        faces=output_faces,
        material_defs=material_defs,
        material_to_texture=material_to_texture,
        texture_cache=texture_cache,
    )

    out_obj_path = obj_path.with_name(obj_path.stem + "_vertexcolor.obj")
    write_vertex_color_obj(
        out_obj_path=out_obj_path,
        header_lines=header_lines,
        out_vertices=out_vertices,
        out_texcoords=out_texcoords,
        out_faces=out_faces,
    )

    print()
    print("========== OUTPUT SUMMARY ==========")
    print(f"Original faces processed: {len(original_faces)}")
    print(f"Output faces written: {len(out_faces)}")
    print(f"Output colored vertices written: {len(out_vertices)}")
    print(f"Output vt written: {len(out_texcoords)}")
    print(f"Face corners colored from textures: {texture_count}")
    print(f"Face corners colored from Kd: {kd_count}")
    print(f"Face corners using fallback color: {fallback_count}")
    print("====================================")
    print()

    print(f"Wrote vertex-color OBJ: {out_obj_path}")
    print("Done.")


if __name__ == "__main__":
    main()
    secondsBeforeExit = 5
    #print(f"Program exits in {secondsBeforeExit} seconds", end="")  # No newline at the end
    for x in range(secondsBeforeExit):
        time.sleep(1)
        print(".", end="")