using System;
using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

public struct Triangle
{
    public float3 v0, v1, v2;
    public float3 uv0, uv1, uv2;
};

public struct MathematicalPlane
{
    public float3 normal;
    public float distance;
};

public struct StartPosition
{
    public float3 playerStart;
    public int sectorId;
};

public struct PolygonMeta
{
    public int edgeStartIndex;
    public int edgeCount;

    public int triangleStartIndex;
    public int triangleCount;

    public int connectedSectorId;
    public int sectorId;

    public int collider;
    public int opaque;

    public int plane;
};

public struct SectorMeta
{
    public int polygonStartIndex;
    public int polygonCount;

    public int planeStartIndex;
    public int planeCount;

    public int sectorId;
};

public struct TrianglesMeta
{
    public int triangleStartIndex;
    public int triangleCount;

    public int planeStartIndex;
    public int planeCount;

    public int sectorId;
};

public struct PortalMeta
{
    public int edgeStartIndex;
    public int edgeCount;

    public int connectedSectorId;
    public int sectorId;

    public int planeStartIndex;
    public int planeCount;

    public int portalContact;
};

[BurstCompile]
public struct SectorsJob : IJobParallelFor
{
    [ReadOnly] public float3 point;
    [ReadOnly] public NativeArray<SectorMeta> currentSectors;
    [ReadOnly] public NativeArray<PolygonMeta> polygons;
    [ReadOnly] public NativeArray<float3> vertices;
    [ReadOnly] public NativeArray<float3> textures;
    [ReadOnly] public NativeArray<int> triangles;
    [ReadOnly] public NativeArray<int> edges;
    [ReadOnly] public NativeArray<SectorMeta> sectors;
    [ReadOnly] public NativeArray<SectorMeta> contains;
    [ReadOnly] public NativeArray<MathematicalPlane> planes;

    public NativeList<TrianglesMeta>.ParallelWriter rawTriangles;
    public NativeList<PortalMeta>.ParallelWriter rawPortals;

    public void Execute(int index)
    {
        SectorMeta sector = currentSectors[index];

        for (int a = sector.polygonStartIndex; a < sector.polygonStartIndex + sector.polygonCount; a++)
        {
            PolygonMeta polygon = polygons[a];

            float planeDistance = math.dot(planes[polygon.plane].normal, point) + planes[polygon.plane].distance;

            if (planeDistance <= 0)
            {
                continue;
            }

            int connectedsector = polygon.connectedSectorId;

            if (connectedsector == -1)
            {
                rawTriangles.AddNoResize(new TrianglesMeta
                {
                    triangleStartIndex = polygon.triangleStartIndex,
                    triangleCount = polygon.triangleCount,

                    planeStartIndex = sector.planeStartIndex,
                    planeCount = sector.planeCount,

                    sectorId = sector.sectorId
                });

                continue;
            }

            SectorMeta sectorpolygon = sectors[connectedsector];

            int contact = 1;

            for (int b = 0; b < contains.Length; b++)
            {
                if (contains[b].sectorId == sectorpolygon.sectorId)
                {
                    contact = 0;
                    break;
                }
            }

            rawPortals.AddNoResize(new PortalMeta
            {
                edgeStartIndex = polygon.edgeStartIndex,
                edgeCount = polygon.edgeCount,

                connectedSectorId = polygon.connectedSectorId,
                sectorId = polygon.sectorId,

                planeStartIndex = sector.planeStartIndex,
                planeCount = sector.planeCount,

                portalContact = contact
            });
        }
    }
}

[BurstCompile]
public struct ClipTrianglesJob : IJobParallelFor
{
    [ReadOnly] public NativeArray<TrianglesMeta> rawTriangles;
    [ReadOnly] public NativeArray<MathematicalPlane> currentFrustums;
    [ReadOnly] public NativeArray<float3> vertices;
    [ReadOnly] public NativeArray<float3> textures;
    [ReadOnly] public NativeArray<int> triangles;

    [NativeDisableParallelForRestriction]
    public NativeArray<float3> processvertices;

    [NativeDisableParallelForRestriction]
    public NativeArray<float3> processtextures;

    [NativeDisableParallelForRestriction]
    public NativeArray<bool> processbool;

    [NativeDisableParallelForRestriction]
    public NativeArray<float3> temporaryvertices;

    [NativeDisableParallelForRestriction]
    public NativeArray<float3> temporarytextures;

    public NativeList<Triangle>.ParallelWriter finalTriangles;

    public void Execute(int index)
    {
        int baseIndex = index * 256;

        TrianglesMeta tm = rawTriangles[index];

        for (int a = tm.triangleStartIndex; a < tm.triangleStartIndex + tm.triangleCount; a += 3)
        {
            int processverticescount = 0;
            int processtexturescount = 0;
            int processboolcount = 0;

            processvertices[baseIndex + processverticescount] = vertices[triangles[a]];
            processvertices[baseIndex + processverticescount + 1] = vertices[triangles[a + 1]];
            processvertices[baseIndex + processverticescount + 2] = vertices[triangles[a + 2]];
            processverticescount += 3;
            processtextures[baseIndex + processtexturescount] = textures[triangles[a]];
            processtextures[baseIndex + processtexturescount + 1] = textures[triangles[a + 1]];
            processtextures[baseIndex + processtexturescount + 2] = textures[triangles[a + 2]];
            processtexturescount += 3;
            processbool[baseIndex + processboolcount] = true;
            processbool[baseIndex + processboolcount + 1] = true;
            processbool[baseIndex + processboolcount + 2] = true;
            processboolcount += 3;

            for (int b = tm.planeStartIndex; b < tm.planeStartIndex + tm.planeCount; b++)
            {
                int addTriangles = 0;

                int temporaryverticescount = 0;
                int temporarytexturescount = 0;

                for (int c = baseIndex; c < baseIndex + processverticescount; c += 3)
                {
                    if (processbool[c] == false && processbool[c + 1] == false && processbool[c + 2] == false)
                    {
                        continue;
                    }

                    float3 v0 = processvertices[c];
                    float3 v1 = processvertices[c + 1];
                    float3 v2 = processvertices[c + 2];

                    float3 uv0 = processtextures[c];
                    float3 uv1 = processtextures[c + 1];
                    float3 uv2 = processtextures[c + 2];

                    float d0 = math.dot(currentFrustums[b].normal, v0) + currentFrustums[b].distance;
                    float d1 = math.dot(currentFrustums[b].normal, v1) + currentFrustums[b].distance;
                    float d2 = math.dot(currentFrustums[b].normal, v2) + currentFrustums[b].distance;

                    bool b0 = d0 >= 0;
                    bool b1 = d1 >= 0;
                    bool b2 = d2 >= 0;

                    if (b0 && b1 && b2)
                    {
                        continue;
                    }
                    else if ((b0 && !b1 && !b2) || (!b0 && b1 && !b2) || (!b0 && !b1 && b2))
                    {
                        float3 inV, outV1, outV2;
                        float3 inUV, outUV1, outUV2;
                        float inD, outD1, outD2;

                        if (b0)
                        {
                            inV = v0;
                            inUV = uv0;
                            inD = d0;
                            outV1 = v1;
                            outUV1 = uv1;
                            outD1 = d1;
                            outV2 = v2;
                            outUV2 = uv2;
                            outD2 = d2;
                        }
                        else if (b1)
                        {
                            inV = v1;
                            inUV = uv1;
                            inD = d1;
                            outV1 = v2;
                            outUV1 = uv2;
                            outD1 = d2;
                            outV2 = v0;
                            outUV2 = uv0;
                            outD2 = d0;
                        }
                        else
                        {
                            inV = v2;
                            inUV = uv2;
                            inD = d2;
                            outV1 = v0;
                            outUV1 = uv0;
                            outD1 = d0;
                            outV2 = v1;
                            outUV2 = uv1;
                            outD2 = d1;
                        }

                        float t1 = inD / (inD - outD1);
                        float t2 = inD / (inD - outD2);

                        temporaryvertices[baseIndex + temporaryverticescount] = inV;
                        temporaryvertices[baseIndex + temporaryverticescount + 1] = math.lerp(inV, outV1, t1);
                        temporaryvertices[baseIndex + temporaryverticescount + 2] = math.lerp(inV, outV2, t2);
                        temporaryverticescount += 3;
                        temporarytextures[baseIndex + temporarytexturescount] = inUV;
                        temporarytextures[baseIndex + temporarytexturescount + 1] = math.lerp(inUV, outUV1, t1);
                        temporarytextures[baseIndex + temporarytexturescount + 2] = math.lerp(inUV, outUV2, t2);
                        temporarytexturescount += 3;
                        processbool[c] = false;
                        processbool[c + 1] = false;
                        processbool[c + 2] = false;

                        addTriangles += 1;
                    }
                    else if ((!b0 && b1 && b2) || (b0 && !b1 && b2) || (b0 && b1 && !b2))
                    {
                        float3 inV1, inV2, outV;
                        float3 inUV1, inUV2, outUV;
                        float inD1, inD2, outD;

                        if (!b0)
                        {
                            outV = v0;
                            outUV = uv0;
                            outD = d0;
                            inV1 = v1;
                            inUV1 = uv1;
                            inD1 = d1;
                            inV2 = v2;
                            inUV2 = uv2;
                            inD2 = d2;
                        }
                        else if (!b1)
                        {
                            outV = v1;
                            outUV = uv1;
                            outD = d1;
                            inV1 = v2;
                            inUV1 = uv2;
                            inD1 = d2;
                            inV2 = v0;
                            inUV2 = uv0;
                            inD2 = d0;
                        }
                        else
                        {
                            outV = v2;
                            outUV = uv2;
                            outD = d2;
                            inV1 = v0;
                            inUV1 = uv0;
                            inD1 = d0;
                            inV2 = v1;
                            inUV2 = uv1;
                            inD2 = d1;
                        }

                        float t1 = inD1 / (inD1 - outD);
                        float t2 = inD2 / (inD2 - outD);

                        float3 vA = math.lerp(inV1, outV, t1);
                        float3 vB = math.lerp(inV2, outV, t2);

                        float3 uvA = math.lerp(inUV1, outUV, t1);
                        float3 uvB = math.lerp(inUV2, outUV, t2);

                        temporaryvertices[baseIndex + temporaryverticescount] = inV1;
                        temporaryvertices[baseIndex + temporaryverticescount + 1] = inV2;
                        temporaryvertices[baseIndex + temporaryverticescount + 2] = vA;
                        temporaryverticescount += 3;
                        temporarytextures[baseIndex + temporarytexturescount] = inUV1;
                        temporarytextures[baseIndex + temporarytexturescount + 1] = inUV2;
                        temporarytextures[baseIndex + temporarytexturescount + 2] = uvA;
                        temporarytexturescount += 3;
                        temporaryvertices[baseIndex + temporaryverticescount] = vA;
                        temporaryvertices[baseIndex + temporaryverticescount + 1] = inV2;
                        temporaryvertices[baseIndex + temporaryverticescount + 2] = vB;
                        temporaryverticescount += 3;
                        temporarytextures[baseIndex + temporarytexturescount] = uvA;
                        temporarytextures[baseIndex + temporarytexturescount + 1] = inUV2;
                        temporarytextures[baseIndex + temporarytexturescount + 2] = uvB;
                        temporarytexturescount += 3;
                        processbool[c] = false;
                        processbool[c + 1] = false;
                        processbool[c + 2] = false;

                        addTriangles += 2;
                    }
                    else
                    {
                        processbool[c] = false;
                        processbool[c + 1] = false;
                        processbool[c + 2] = false;
                    }
                }

                if (addTriangles > 0)
                {
                    for (int d = baseIndex; d < baseIndex + temporaryverticescount; d += 3)
                    {
                        processvertices[baseIndex + processverticescount] = temporaryvertices[d];
                        processvertices[baseIndex + processverticescount + 1] = temporaryvertices[d + 1];
                        processvertices[baseIndex + processverticescount + 2] = temporaryvertices[d + 2];
                        processverticescount += 3;
                        processtextures[baseIndex + processtexturescount] = temporarytextures[d];
                        processtextures[baseIndex + processtexturescount + 1] = temporarytextures[d + 1];
                        processtextures[baseIndex + processtexturescount + 2] = temporarytextures[d + 2];
                        processtexturescount += 3;
                        processbool[baseIndex + processboolcount] = true;
                        processbool[baseIndex + processboolcount + 1] = true;
                        processbool[baseIndex + processboolcount + 2] = true;
                        processboolcount += 3;
                    }
                }
            }

            for (int e = baseIndex; e < baseIndex + processverticescount; e += 3)
            {
                if (processbool[e] == true && processbool[e + 1] == true && processbool[e + 2] == true)
                {
                    finalTriangles.AddNoResize(new Triangle
                    {
                        v0 = processvertices[e],
                        v1 = processvertices[e + 1],
                        v2 = processvertices[e + 2],
                        uv0 = processtextures[e],
                        uv1 = processtextures[e + 1],
                        uv2 = processtextures[e + 2]
                    });
                }
            }
        }
    }
}

[BurstCompile]
public struct ClipPortalsJob : IJobParallelFor
{
    [ReadOnly] public NativeArray<PortalMeta> rawPortals;
    [ReadOnly] public NativeArray<MathematicalPlane> currentFrustums;
    [ReadOnly] public NativeArray<MathematicalPlane> originalFrustum;
    [ReadOnly] public NativeArray<float3> vertices;
    [ReadOnly] public NativeArray<int> edges;
    [ReadOnly] public NativeArray<SectorMeta> sectors;
    [ReadOnly] public float3 point;

    [NativeDisableParallelForRestriction]
    public NativeArray<float3> outedges;

    [NativeDisableParallelForRestriction]
    public NativeArray<float3> processedgevertices;

    [NativeDisableParallelForRestriction]
    public NativeArray<bool> processedgebool;

    [NativeDisableParallelForRestriction]
    public NativeArray<float3> temporaryedgevertices;

    [NativeDisableParallelForRestriction]
    public NativeArray<MathematicalPlane> nextFrustums;

    public NativeList<SectorMeta>.ParallelWriter nextSectors;

    public void Execute(int index)
    {
        int baseIndex = index * 256;

        PortalMeta portal = rawPortals[index];

        int connectedsector = portal.connectedSectorId;

        SectorMeta sectorportal = sectors[connectedsector];

        int connectedstart = sectorportal.polygonStartIndex;
        int connectedcount = sectorportal.polygonCount;

        if (portal.portalContact == 0)
        {
            int contactIndex = baseIndex;

            nextFrustums[contactIndex] = originalFrustum[0];
            nextFrustums[contactIndex + 1] = originalFrustum[1];
            nextFrustums[contactIndex + 2] = originalFrustum[2];
            nextFrustums[contactIndex + 3] = originalFrustum[3];

            nextSectors.AddNoResize(new SectorMeta
            {
                polygonStartIndex = connectedstart,
                polygonCount = connectedcount,
                planeStartIndex = contactIndex,
                planeCount = originalFrustum.Length,
                sectorId = connectedsector
            });

            return;
        }

        int outedgescount = 0;
        int processedgescount = 0;
        int processedgesboolcount = 0;

        for (int a = portal.edgeStartIndex; a < portal.edgeStartIndex + portal.edgeCount; a += 2)
        {
            processedgevertices[baseIndex + processedgescount] = vertices[edges[a]];
            processedgevertices[baseIndex + processedgescount + 1] = vertices[edges[a + 1]];
            processedgescount += 2;
            processedgebool[baseIndex + processedgesboolcount] = true;
            processedgebool[baseIndex + processedgesboolcount + 1] = true;
            processedgesboolcount += 2;
        }

        for (int b = portal.planeStartIndex; b < portal.planeStartIndex + portal.planeCount; b++)
        {
            int intersection = 0;
            int temporaryverticescount = 0;

            float3 intersectionPoint1 = float3.zero;
            float3 intersectionPoint2 = float3.zero;

            for (int c = baseIndex; c < baseIndex + processedgescount; c += 2)
            {
                if (processedgebool[c] == false && processedgebool[c + 1] == false)
                {
                    continue;
                }

                float3 p1 = processedgevertices[c];
                float3 p2 = processedgevertices[c + 1];

                float d1 = math.dot(currentFrustums[b].normal, p1) + currentFrustums[b].distance;
                float d2 = math.dot(currentFrustums[b].normal, p2) + currentFrustums[b].distance;

                bool b0 = d1 >= 0;
                bool b1 = d2 >= 0;

                if (b0 && b1)
                {
                    continue;
                }
                else if ((b0 && !b1) || (!b0 && b1))
                {
                    float3 point1;
                    float3 point2;

                    float t = d1 / (d1 - d2);

                    float3 intersectionPoint = math.lerp(p1, p2, t);

                    if (b0)
                    {
                        point1 = p1;
                        point2 = intersectionPoint;
                        intersectionPoint1 = intersectionPoint;
                    }
                    else
                    {
                        point1 = intersectionPoint;
                        point2 = p2;
                        intersectionPoint2 = intersectionPoint;
                    }

                    temporaryedgevertices[baseIndex + temporaryverticescount] = point1;
                    temporaryedgevertices[baseIndex + temporaryverticescount + 1] = point2;
                    temporaryverticescount += 2;

                    processedgebool[c] = false;
                    processedgebool[c + 1] = false;

                    intersection += 1;
                }
                else
                {
                    processedgebool[c] = false;
                    processedgebool[c + 1] = false;
                }
            }

            if (intersection == 2)
            {
                for (int d = baseIndex; d < baseIndex + temporaryverticescount; d += 2)
                {
                    processedgevertices[baseIndex + processedgescount] = temporaryedgevertices[d];
                    processedgevertices[baseIndex + processedgescount + 1] = temporaryedgevertices[d + 1];
                    processedgescount += 2;
                    processedgebool[baseIndex + processedgesboolcount] = true;
                    processedgebool[baseIndex + processedgesboolcount + 1] = true;
                    processedgesboolcount += 2;
                }

                processedgevertices[baseIndex + processedgescount] = intersectionPoint1;
                processedgevertices[baseIndex + processedgescount + 1] = intersectionPoint2;
                processedgescount += 2;
                processedgebool[baseIndex + processedgesboolcount] = true;
                processedgebool[baseIndex + processedgesboolcount + 1] = true;
                processedgesboolcount += 2;
            }
        }

        for (int e = baseIndex; e < baseIndex + processedgescount; e += 2)
        {
            if (processedgebool[e] == true && processedgebool[e + 1] == true)
            {
                outedges[baseIndex + outedgescount] = processedgevertices[e];
                outedges[baseIndex + outedgescount + 1] = processedgevertices[e + 1];
                outedgescount += 2;
            }
        }

        if (outedgescount < 6 || outedgescount % 2 == 1)
        {
            return;
        }

        int StartIndex = baseIndex;

        int IndexCount = 0;

        for (int f = baseIndex; f < baseIndex + outedgescount; f += 2)
        {
            float3 p0 = outedges[f];
            float3 p1 = outedges[f + 1];
            float3 normal = math.cross(p0 - p1, point - p1);
            float magnitude = math.length(normal);

            if (magnitude < 0.01f)
            {
                continue;
            }

            float3 normalized = normal / magnitude;

            float distance = -math.dot(normalized, p0);

            nextFrustums[StartIndex + IndexCount] = new MathematicalPlane { normal = normalized, distance = distance };
            IndexCount += 1;
        }

        nextSectors.AddNoResize(new SectorMeta
        {
            polygonStartIndex = connectedstart,
            polygonCount = connectedcount,
            planeStartIndex = StartIndex,
            planeCount = IndexCount,
            sectorId = connectedsector
        });
    }
}

public class LevelLoader : MonoBehaviour
{
    public string Name = "twohallways-clear";

    public float speed = 7f;
    public float jumpHeight = 2f;
    public float gravity = 5f;
    public float sensitivity = 10f;
    public float clampAngle = 90f;
    public float smoothFactor = 25f;

    private Vector2 targetRotation;
    private Vector3 targetMovement;
    private Vector2 currentRotation;
    private Vector3 currentForce;

    private CharacterController Player;

    private TopLevelLists LevelLists;
    private List<Vector2> vertices = new List<Vector2>();
    private List<Sector> sectors = new List<Sector>();
    private List<StartSector> starts = new List<StartSector>();
    private List<Vector3> ceilingverts = new List<Vector3>();
    private List<int> ceilingtri = new List<int>();
    private List<Vector3> floorverts = new List<Vector3>();
    private List<int> floortri = new List<int>();
    private Material opaquematerial;
    private List<MeshCollider> CollisionSectors = new List<MeshCollider>();
    private List<Vector3> OpaqueVertices = new List<Vector3>();
    private List<int> OpaqueTriangles = new List<int>();
    private List<Mesh> CollisionMesh = new List<Mesh>();
    private GameObject CollisionObjects;
    private NativeArray<bool> processbool;
    private NativeArray<float3> processvertices;
    private NativeArray<float3> processtextures;
    private NativeArray<float3> temporaryvertices;
    private NativeArray<float3> temporarytextures;
    private NativeArray<float3> outEdges;
    private NativeArray<float3> processedgevertices;
    private NativeArray<bool> processedgebool;
    private NativeArray<float3> temporaryedgevertices;
    private NativeArray<MathematicalPlane> planeA;
    private NativeArray<MathematicalPlane> planeB;
    private NativeList<SectorMeta> sideA;
    private NativeList<SectorMeta> sideB;
    private NativeList<Triangle> outTriangles;
    private NativeList<TrianglesMeta> rawTriangles;
    private NativeList<PortalMeta> rawPortals;
    private NativeList<SectorMeta> contains;
    private NativeList<SectorMeta> oldContains;
    private NativeList<MathematicalPlane> OriginalFrustum;

    private List<List<SectorMeta>> ListOfSectorLists = new List<List<SectorMeta>>();
    private Camera Cam;
    private Vector3 CamPoint;
    private SectorMeta CurrentSector;
    private bool radius;
    private bool check;
    private double Ceiling;
    private double Floor;
    private MathematicalPlane LeftPlane;
    private MathematicalPlane TopPlane;
    private List<Vector3> flooruvs = new List<Vector3>();
    private List<Vector3> ceilinguvs = new List<Vector3>();
    private GraphicsBuffer triBuffer;


    [Serializable]
    public class Sector
    {
        public float floorHeight;
        public float ceilingHeight;
        public List<int> vertexIndices = new List<int>();
        public List<int> wallTypes = new List<int>(); // -1 for solid, sector index for portal
    }

    [Serializable]
    public class StartSector
    {
        public Vector3 location;
        public float angle;
        public int sector;
    }

    [Serializable]
    public class TopLevelLists
    {
        public NativeList<float3> vertices;
        public NativeList<float3> textures;
        public NativeList<int> triangles;
        public NativeList<int> edges;
        public NativeList<MathematicalPlane> planes;
        public NativeList<PolygonMeta> polygons;
        public NativeList<SectorMeta> sectors;
        public NativeList<StartPosition> positions;
    }

    void Start()
    {
        int strideTriangle = System.Runtime.InteropServices.Marshal.SizeOf(typeof(Triangle));

        triBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, (LevelLists.sectors.Length * 32) * 128, strideTriangle);

        opaquematerial.SetBuffer("outputTriangleBuffer", triBuffer);

        for (int i = 0; i < 2; i++)
        {
            ListOfSectorLists.Add(new List<SectorMeta>());
        }

        for (int i = 0; i < LevelLists.sectors.Length; i++)
        {
            Physics.IgnoreCollision(Player, CollisionSectors[LevelLists.sectors[i].sectorId], true);
        }
    }

    void Update()
    {
        PlayerInput();

        if (Cam.transform.hasChanged)
        {
            CamPoint = Cam.transform.position;

            GetSectors(CurrentSector);

            OriginalFrustum.Clear();

            ReadFrustumPlanes(Cam, OriginalFrustum);

            OriginalFrustum.RemoveAt(5);

            OriginalFrustum.RemoveAt(4);

            GetPolygons(CurrentSector);

            Cam.transform.hasChanged = false;
        }
    }

    void Awake()
    {
        Player = GameObject.Find("Player").GetComponent<CharacterController>();

        Player.GetComponent<CharacterController>().enabled = true;

        Cursor.lockState = CursorLockMode.Locked;

        Cam = Camera.main;

        LevelLists = new TopLevelLists();

        LevelLists.edges = new NativeList<int>(Allocator.Persistent);
        LevelLists.triangles = new NativeList<int>(Allocator.Persistent);
        LevelLists.vertices = new NativeList<float3>(Allocator.Persistent);
        LevelLists.textures = new NativeList<float3>(Allocator.Persistent);
        LevelLists.sectors = new NativeList<SectorMeta>(Allocator.Persistent);
        LevelLists.planes = new NativeList<MathematicalPlane>(Allocator.Persistent);
        LevelLists.polygons = new NativeList<PolygonMeta>(Allocator.Persistent);
        LevelLists.positions = new NativeList<StartPosition>(Allocator.Persistent);

        CollisionObjects = new GameObject("Collision Meshes");

        LoadFromFile();

        CreateMaterial();

        BuildGeometry();

        BuildObjects();

        BuildColliders();

        PlayerStart();

        processbool = new NativeArray<bool>((LevelLists.sectors.Length * 32) * 256, Allocator.Persistent);
        processvertices = new NativeArray<float3>((LevelLists.sectors.Length * 32) * 256, Allocator.Persistent);
        processtextures = new NativeArray<float3>((LevelLists.sectors.Length * 32) * 256, Allocator.Persistent);
        temporaryvertices = new NativeArray<float3>((LevelLists.sectors.Length * 32) * 256, Allocator.Persistent);
        temporarytextures = new NativeArray<float3>((LevelLists.sectors.Length * 32) * 256, Allocator.Persistent);
        processedgebool = new NativeArray<bool>((LevelLists.sectors.Length * 32) * 256, Allocator.Persistent);
        processedgevertices = new NativeArray<float3>((LevelLists.sectors.Length * 32) * 256, Allocator.Persistent);
        temporaryedgevertices = new NativeArray<float3>((LevelLists.sectors.Length * 32) * 256, Allocator.Persistent);
        outEdges = new NativeArray<float3>((LevelLists.sectors.Length * 32) * 256, Allocator.Persistent);
        planeA = new NativeArray<MathematicalPlane>((LevelLists.sectors.Length * 32) * 256, Allocator.Persistent);
        planeB = new NativeArray<MathematicalPlane>((LevelLists.sectors.Length * 32) * 256, Allocator.Persistent);
        contains = new NativeList<SectorMeta>(Allocator.Persistent);
        oldContains = new NativeList<SectorMeta>(Allocator.Persistent);
        sideA = new NativeList<SectorMeta>(LevelLists.sectors.Length * 32, Allocator.Persistent);
        sideB = new NativeList<SectorMeta>(LevelLists.sectors.Length * 32, Allocator.Persistent);
        outTriangles = new NativeList<Triangle>((LevelLists.sectors.Length * 32) * 128, Allocator.Persistent);
        OriginalFrustum = new NativeList<MathematicalPlane>(6, Allocator.Persistent);
        rawTriangles = new NativeList<TrianglesMeta>(LevelLists.polygons.Length * 32, Allocator.Persistent);
        rawPortals = new NativeList<PortalMeta>(LevelLists.polygons.Length * 32, Allocator.Persistent);
    }

void FixedUpdate()
    {
        if (!Player.isGrounded)
        {
            currentForce.y -= gravity * Time.deltaTime;
        }
    }

    void OnDestroy()
    {
        triBuffer?.Dispose();

        if (LevelLists.sectors.IsCreated)
        {
            LevelLists.sectors.Dispose();
        }
        if (LevelLists.polygons.IsCreated)
        {
            LevelLists.polygons.Dispose();
        }
        if (LevelLists.vertices.IsCreated)
        {
            LevelLists.vertices.Dispose();
        }
        if (LevelLists.textures.IsCreated)
        {
            LevelLists.textures.Dispose();
        }
        if (LevelLists.triangles.IsCreated)
        {
            LevelLists.triangles.Dispose();
        }
        if (LevelLists.edges.IsCreated)
        {
            LevelLists.edges.Dispose();
        }
        if (LevelLists.positions.IsCreated)
        {
            LevelLists.positions.Dispose();
        }
        if (LevelLists.planes.IsCreated)
        {
            LevelLists.planes.Dispose();
        }
        if (contains.IsCreated)
        {
            contains.Dispose();
        }
        if (processbool.IsCreated)
        {
            processbool.Dispose();
        }
        if (processvertices.IsCreated)
        {
            processvertices.Dispose();
        }
        if (processtextures.IsCreated)
        {
            processtextures.Dispose();
        }
        if (temporaryvertices.IsCreated)
        {
            temporaryvertices.Dispose();
        }
        if (temporarytextures.IsCreated)
        {
            temporarytextures.Dispose();
        }
        if (outEdges.IsCreated)
        {
            outEdges.Dispose();
        }
        if (planeA.IsCreated)
        {
            planeA.Dispose();
        }
        if (planeB.IsCreated)
        {
            planeB.Dispose();
        }
        if (sideA.IsCreated)
        {
            sideA.Dispose();
        }
        if (sideB.IsCreated)
        {
            sideB.Dispose();
        }
        if (outTriangles.IsCreated)
        {
            outTriangles.Dispose();
        }
        if (OriginalFrustum.IsCreated)
        {
            OriginalFrustum.Dispose();
        }
        if (oldContains.IsCreated)
        {
            oldContains.Dispose();
        }
        if (rawTriangles.IsCreated)
        {
            rawTriangles.Dispose();
        }
        if (rawPortals.IsCreated)
        {
            rawPortals.Dispose();
        }
        if (processedgevertices.IsCreated)
        {
            processedgevertices.Dispose();
        }
        if (temporaryedgevertices.IsCreated)
        {
            temporaryedgevertices.Dispose();
        }
        if (processedgebool.IsCreated)
        {
            processedgebool.Dispose();
        }
    }

    void OnRenderObject()
    {
        triBuffer.SetData(outTriangles.AsArray());

        opaquematerial.SetPass(0);
        Graphics.DrawProceduralNow(MeshTopology.Triangles, outTriangles.Length * 3);
    }

    public void CreateMaterial()
    {
        Shader shader = Resources.Load<Shader>("TriangleTexArray");

        opaquematerial = new Material(shader);

        opaquematerial.mainTexture = Resources.Load<Texture2DArray>("Textures");
    }

    private MathematicalPlane FromVec4(Vector4 aVec)
    {
        Vector3 n = new Vector3(aVec.x, aVec.y, aVec.z);
        float l = n.magnitude;
        return new MathematicalPlane
        {
            normal = n / l,
            distance = aVec.w / l
        };
    }

    public void SetFrustumPlanes(NativeList<MathematicalPlane> planes, Matrix4x4 m)
    {   
        var r0 = m.GetRow(0);
        var r1 = m.GetRow(1);
        var r2 = m.GetRow(2);
        var r3 = m.GetRow(3);

        planes.Add(FromVec4(r3 - r0)); // Right
        planes.Add(FromVec4(r3 + r0)); // Left
        planes.Add(FromVec4(r3 - r1)); // Top
        planes.Add(FromVec4(r3 + r1)); // Bottom
        planes.Add(FromVec4(r3 - r2)); // Far
        planes.Add(FromVec4(r3 + r2)); // Near
    }

    public void ReadFrustumPlanes(Camera cam, NativeList<MathematicalPlane> planes)
    {
        SetFrustumPlanes(planes, cam.projectionMatrix * cam.worldToCameraMatrix);
    }

    public void PlayerInput()
    {
        if (Input.GetKeyDown(KeyCode.Escape))
        {
            Application.Quit();
        }
        if (Input.GetKeyDown(KeyCode.Space) && Player.isGrounded)
        {
            currentForce.y = jumpHeight;
        }

        float mousex = Input.GetAxisRaw("Mouse X");
        float mousey = Input.GetAxisRaw("Mouse Y");

        targetRotation.x -= mousey * sensitivity;
        targetRotation.y += mousex * sensitivity;

        targetRotation.x = Mathf.Clamp(targetRotation.x, -clampAngle, clampAngle);

        currentRotation = Vector2.Lerp(currentRotation, targetRotation, smoothFactor * Time.deltaTime);

        Cam.transform.localRotation = Quaternion.Euler(currentRotation.x, 0f, 0f);
        Player.transform.rotation = Quaternion.Euler(0f, currentRotation.y, 0f);

        float horizontal = Input.GetAxisRaw("Horizontal");
        float vertical = Input.GetAxisRaw("Vertical");

        targetMovement = (Player.transform.right * horizontal + Player.transform.forward * vertical).normalized;

        Player.Move((targetMovement + currentForce) * speed * Time.deltaTime);
    }

    public float GetPlaneSignedDistanceToPoint(MathematicalPlane plane, Vector3 point)
    {
        return Vector3.Dot(plane.normal, point) + plane.distance;
    }

    public bool CheckRadius(SectorMeta asector, Vector3 campoint)
    {
        for (int i = asector.polygonStartIndex; i < asector.polygonStartIndex + asector.polygonCount; i++)
        {
            if (GetPlaneSignedDistanceToPoint(LevelLists.planes[LevelLists.polygons[i].plane], campoint) < -0.6f)
            {
                return false;
            }
        }
        return true;
    }

    public bool CheckSector(SectorMeta asector, Vector3 campoint)
    {
        for (int i = asector.polygonStartIndex; i < asector.polygonStartIndex + asector.polygonCount; i++)
        {
            if (GetPlaneSignedDistanceToPoint(LevelLists.planes[LevelLists.polygons[i].plane], campoint) < 0)
            {
                return false;
            }
        }
        return true;
    }

    public bool SectorsContains(int sectorID)
    {
        for (int i = 0; i < contains.Length; i++)
        {
            if (contains[i].sectorId == sectorID)
            {
                return true;
            }
        }
        return false;
    }

    public bool SectorsDoNotEqual()
    {
        if (contains.Length != oldContains.Length)
        {
            return true;
        }

        for (int i = 0; i < contains.Length; i++)
        {
            if (contains[i].sectorId != oldContains[i].sectorId)
            {
                return true;
            }
        }
        return false;
    }

    public void GetSectors(SectorMeta ASector)
    {
        int input = 0;
        int output = 1;

        contains.Clear();

        ListOfSectorLists[input].Clear();
        ListOfSectorLists[output].Clear();

        ListOfSectorLists[input].Add(ASector);

        for (int a = 0; a < oldContains.Length; a++)
        {
            Physics.IgnoreCollision(Player, CollisionSectors[oldContains[a].sectorId], true);
        }

        for (int b = 0; b < 4096; b++)
        {
            if (b % 2 == 0)
            {
                input = 0;
                output = 1;
            }
            else
            {
                input = 1;
                output = 0;
            }

            ListOfSectorLists[output].Clear();

            if (ListOfSectorLists[input].Count == 0)
            {
                break;
            }

            for (int c = 0; c < ListOfSectorLists[input].Count; c++)
            {
                SectorMeta sector = ListOfSectorLists[input][c];

                contains.Add(sector);

                Physics.IgnoreCollision(Player, CollisionSectors[sector.sectorId], false);

                for (int d = sector.polygonStartIndex; d < sector.polygonStartIndex + sector.polygonCount; d++)
                {
                    int connectedsector = LevelLists.polygons[d].connectedSectorId;

                    if (connectedsector == -1)
                    {
                        continue;
                    }

                    SectorMeta portalsector = LevelLists.sectors[connectedsector];

                    if (SectorsContains(portalsector.sectorId))
                    {
                        continue;
                    }

                    radius = CheckRadius(portalsector, CamPoint);

                    if (radius)
                    {
                        ListOfSectorLists[output].Add(portalsector);
                    }
                }

                check = CheckSector(sector, CamPoint);

                if (check)
                {
                    CurrentSector = sector;
                }
            }    
        }

        if (SectorsDoNotEqual())
        {
            oldContains.Clear();

            for (int e = 0; e < contains.Length; e++)
            {
                oldContains.Add(contains[e]);
            }
        }
    }

    public void GetPolygons(SectorMeta ASector)
    {
        sideA.Clear();
        sideB.Clear();
        outTriangles.Clear();

        int jobsCompleted = 0;

        planeA[0] = OriginalFrustum[0];
        planeA[1] = OriginalFrustum[1];
        planeA[2] = OriginalFrustum[2];
        planeA[3] = OriginalFrustum[3];

        sideA.Add(ASector);

        NativeList<SectorMeta> currentSectors = sideA;
        NativeList<SectorMeta> nextSectors = sideB;
        NativeArray<MathematicalPlane> currentFrustums = planeA;
        NativeArray<MathematicalPlane> nextFrustums = planeB;

        while (currentSectors.Length > 0)
        {
            nextSectors.Clear();
            rawTriangles.Clear();
            rawPortals.Clear();

            JobHandle h1 = new SectorsJob
            {
                point = CamPoint,
                vertices = LevelLists.vertices.AsDeferredJobArray(),
                textures = LevelLists.textures.AsDeferredJobArray(),
                triangles = LevelLists.triangles.AsDeferredJobArray(),
                edges = LevelLists.edges.AsDeferredJobArray(),
                planes = LevelLists.planes.AsDeferredJobArray(),
                polygons = LevelLists.polygons.AsDeferredJobArray(),
                contains = contains.AsDeferredJobArray(),
                sectors = LevelLists.sectors.AsDeferredJobArray(),
                currentSectors = currentSectors.AsDeferredJobArray(),
                rawPortals = rawPortals.AsParallelWriter(),
                rawTriangles = rawTriangles.AsParallelWriter()
            }.Schedule(currentSectors.Length, 32);

            h1.Complete();

            JobHandle h2 = new ClipTrianglesJob 
            {
                rawTriangles = rawTriangles.AsDeferredJobArray(),
                vertices = LevelLists.vertices.AsDeferredJobArray(),
                textures = LevelLists.textures.AsDeferredJobArray(),
                triangles = LevelLists.triangles.AsDeferredJobArray(),
                currentFrustums = currentFrustums,
                processvertices = processvertices,
                processtextures = processtextures,
                processbool = processbool,
                temporaryvertices = temporaryvertices,
                temporarytextures = temporarytextures,
                finalTriangles = outTriangles.AsParallelWriter()
            }.Schedule(rawTriangles.Length, 64);

            JobHandle h3 = new ClipPortalsJob 
            {
                rawPortals = rawPortals.AsDeferredJobArray(),
                point = CamPoint,
                vertices = LevelLists.vertices.AsDeferredJobArray(),
                originalFrustum = OriginalFrustum.AsDeferredJobArray(),
                outedges = outEdges,
                edges = LevelLists.edges.AsDeferredJobArray(),
                sectors = LevelLists.sectors.AsDeferredJobArray(),
                processedgebool = processedgebool,
                temporaryedgevertices = temporaryedgevertices,
                processedgevertices = processedgevertices,
                currentFrustums = currentFrustums,
                nextFrustums = nextFrustums,
                nextSectors = nextSectors.AsParallelWriter()
            }.Schedule(rawPortals.Length, 64);

            JobHandle.CombineDependencies(h2, h3).Complete();

            jobsCompleted += 1;

            if (jobsCompleted % 2 == 0)
            {
                currentSectors = sideA;
                nextSectors = sideB;
                currentFrustums = planeA;
                nextFrustums = planeB;
            }
            else
            {
                currentSectors = sideB;
                nextSectors = sideA;
                currentFrustums = planeB;
                nextFrustums = planeA;
            }
        }
    }

    public void PlayerStart()
    {
        if (LevelLists.positions.Length == 0)
        {
            Debug.LogError("No player starts available.");

            return;
        }

        int randomIndex = UnityEngine.Random.Range(0, LevelLists.positions.Length);

        StartPosition selectedPosition = LevelLists.positions[randomIndex];

        CurrentSector = LevelLists.sectors[selectedPosition.sectorId];

        Player.transform.position = new Vector3(selectedPosition.playerStart.z, selectedPosition.playerStart.y + 1.10f, selectedPosition.playerStart.x);
    }

    public void LoadFromFile()
    {
        TextAsset file = Resources.Load<TextAsset>(Name);
        if (file == null)
        {
            Debug.LogError("File not found in Resources!");
            return;
        }

        string[] lines = file.text.Split('\n');

        for (int i = 0; i < lines.Length; i++)
        {
            if (lines[i].StartsWith("vertex"))
            {
                string[] parts = lines[i].Split('\t');

                if (parts.Length == 3)
                {
                    float y = float.Parse(parts[1]);

                    string[] xValues = parts[2].Split(' ');

                    for (int e = 0; e < xValues.Length; e++)
                    {
                        if (float.TryParse(xValues[e], out float x))
                        {
                            vertices.Add(new Vector2(x, y));
                        }
                    }
                }
            }

            if (lines[i].StartsWith("sector"))
            {
                Sector sector = new Sector();

                string[] parts = lines[i].Split('\t');

                if (parts.Length == 3)
                {
                    string[] heightParts = parts[1].Split(' ');

                    if (heightParts.Length == 2)
                    {
                        sector.floorHeight = float.Parse(heightParts[0]);

                        sector.ceilingHeight = float.Parse(heightParts[1]);
                    }

                    string[] values = parts[2].Split(' ');

                    int half = values.Length / 2;

                    for (int e = 0; e < values.Length; e++)
                    {
                        if (int.TryParse(values[e], out int val))
                        {
                            if (e < half)
                            {
                                sector.vertexIndices.Add(val);
                            }
                            else
                            {
                                sector.wallTypes.Add(val);
                            }
                        }
                    }
                }

                sectors.Add(sector);
            }

            if (lines[i].StartsWith("player"))
            {
                StartSector start = new StartSector();

                string[] parts = lines[i].Split('\t');

                if (parts.Length == 4)
                {
                    string[] locationParts = parts[1].Split(' ');

                    if (locationParts.Length == 2)
                    {
                        float x = float.Parse(locationParts[0]);

                        float y = float.Parse(locationParts[1]);

                        start.location = new Vector2(x, y);
                    }

                    start.angle = float.Parse(parts[2]);

                    start.sector = int.Parse(parts[3]);
                }

                starts.Add(start);
            }
        }

        Debug.Log($"Loaded {vertices.Count} vertices.");

        Debug.Log($"Loaded {sectors.Count} sectors.");

        Debug.Log($"Player start: location={starts[0].location}, angle={starts[0].angle}, sector={starts[0].sector}");
    }

    public void BuildGeometry()
    {
        int polygonStart = 0;

        for (int i = 0; i < sectors.Count; i++)
        {
            int polygonCount = 0;

            Sector sector = sectors[i];

            for (int e = 0; e < sector.vertexIndices.Count; e++)
            {
                int current = sector.vertexIndices[e];
                int next = sector.vertexIndices[(e + 1) % sector.vertexIndices.Count];

                int wall = sector.wallTypes[(e + 1) % sector.wallTypes.Count];

                double X1 = vertices[current].x / 2 * 2.5f;
                double Z1 = vertices[current].y / 2 * 2.5f;

                double X0 = vertices[next].x / 2 * 2.5f;
                double Z0 = vertices[next].y / 2 * 2.5f;

                if (wall == -1)
                {
                    double V0 = sector.floorHeight / 8 * 2.5f;
                    double V1 = sector.ceilingHeight / 8 * 2.5f;

                    int baseVert = LevelLists.vertices.Length;

                    int baseStartIndex = LevelLists.triangles.Length;

                    LevelLists.vertices.Add(new float3((float)Z1, (float)V0, (float)X1));
                    LevelLists.vertices.Add(new float3((float)Z1, (float)V1, (float)X1));
                    LevelLists.vertices.Add(new float3((float)Z0, (float)V1, (float)X0));
                    LevelLists.vertices.Add(new float3((float)Z0, (float)V0, (float)X0));

                    LevelLists.triangles.Add(baseVert);
                    LevelLists.triangles.Add(baseVert + 1);
                    LevelLists.triangles.Add(baseVert + 2);
                    LevelLists.triangles.Add(baseVert);
                    LevelLists.triangles.Add(baseVert + 2);
                    LevelLists.triangles.Add(baseVert + 3);

                    float3 v0 = LevelLists.vertices[baseVert];
                    float3 v1 = LevelLists.vertices[baseVert + 1];
                    float3 v2 = LevelLists.vertices[baseVert + 2];

                    float3 n = math.normalize(math.cross(v1 - v0, v2 - v0));

                    float3 leftPlaneNormal = math.normalize(v2 - v1);
                    float leftPlaneDistance = -math.dot(leftPlaneNormal, v1);

                    Vector3 topPlaneNormal = math.normalize(v1 - v0);
                    float topPlaneDistance = -math.dot(topPlaneNormal, v1);

                    LeftPlane = new MathematicalPlane { normal = leftPlaneNormal, distance = leftPlaneDistance };
                    TopPlane = new MathematicalPlane { normal = topPlaneNormal, distance = topPlaneDistance };

                    LevelLists.textures.Add(new float3(GetPlaneSignedDistanceToPoint(LeftPlane, LevelLists.vertices[baseVert]) / 2.5f, GetPlaneSignedDistanceToPoint(TopPlane, LevelLists.vertices[baseVert]) / 2.5f, 3));
                    LevelLists.textures.Add(new float3(GetPlaneSignedDistanceToPoint(LeftPlane, LevelLists.vertices[baseVert + 1]) / 2.5f, GetPlaneSignedDistanceToPoint(TopPlane, LevelLists.vertices[baseVert + 1]) / 2.5f, 3));
                    LevelLists.textures.Add(new float3(GetPlaneSignedDistanceToPoint(LeftPlane, LevelLists.vertices[baseVert + 2]) / 2.5f, GetPlaneSignedDistanceToPoint(TopPlane, LevelLists.vertices[baseVert + 2]) / 2.5f, 3));
                    LevelLists.textures.Add(new float3(GetPlaneSignedDistanceToPoint(LeftPlane, LevelLists.vertices[baseVert + 3]) / 2.5f, GetPlaneSignedDistanceToPoint(TopPlane, LevelLists.vertices[baseVert + 3]) / 2.5f, 3));

                    PolygonMeta transformedmesh = new PolygonMeta
                    {
                        plane = LevelLists.planes.Length,

                        collider = i,

                        opaque = i,

                        sectorId = i,

                        connectedSectorId = -1,

                        edgeStartIndex = -1,

                        edgeCount = -1,

                        triangleStartIndex = baseStartIndex,

                        triangleCount = 6
                    };

                    LevelLists.polygons.Add(transformedmesh);

                    MathematicalPlane plane = new MathematicalPlane
                    {
                        normal = n,
                        distance = -math.dot(n, v0)
                    };

                    LevelLists.planes.Add(plane);

                    polygonCount += 1;
                }
                else
                {
                    if (sector.ceilingHeight > sectors[wall].ceilingHeight)
                    {
                        if (sector.floorHeight < sectors[wall].ceilingHeight)
                        {
                            double C0 = sector.ceilingHeight / 8 * 2.5f;

                            if (sector.ceilingHeight > sectors[wall].ceilingHeight)
                            {
                                Ceiling = sectors[wall].ceilingHeight / 8 * 2.5f;
                            }
                            else
                            {
                                Ceiling = sector.ceilingHeight / 8 * 2.5f;
                            }

                            int baseVert = LevelLists.vertices.Length;

                            int baseStartIndex = LevelLists.triangles.Length;

                            LevelLists.vertices.Add(new float3((float)Z1, (float)Ceiling, (float)X1));
                            LevelLists.vertices.Add(new float3((float)Z1, (float)C0, (float)X1));
                            LevelLists.vertices.Add(new float3((float)Z0, (float)C0, (float)X0));
                            LevelLists.vertices.Add(new float3((float)Z0, (float)Ceiling, (float)X0));

                            LevelLists.triangles.Add(baseVert);
                            LevelLists.triangles.Add(baseVert + 1);
                            LevelLists.triangles.Add(baseVert + 2);
                            LevelLists.triangles.Add(baseVert);
                            LevelLists.triangles.Add(baseVert + 2);
                            LevelLists.triangles.Add(baseVert + 3);

                            float3 v0 = LevelLists.vertices[baseVert];
                            float3 v1 = LevelLists.vertices[baseVert + 1];
                            float3 v2 = LevelLists.vertices[baseVert + 2];

                            float3 n = math.normalize(math.cross(v1 - v0, v2 - v0));

                            float3 leftPlaneNormal = math.normalize(v2 - v1);
                            float leftPlaneDistance = -math.dot(leftPlaneNormal, v1);

                            float3 topPlaneNormal = math.normalize(v1 - v0);
                            float topPlaneDistance = -math.dot(topPlaneNormal, v1);

                            LeftPlane = new MathematicalPlane { normal = leftPlaneNormal, distance = leftPlaneDistance };
                            TopPlane = new MathematicalPlane { normal = topPlaneNormal, distance = topPlaneDistance };

                            LevelLists.textures.Add(new float3(GetPlaneSignedDistanceToPoint(LeftPlane, LevelLists.vertices[baseVert]) / 2.5f, GetPlaneSignedDistanceToPoint(TopPlane, LevelLists.vertices[baseVert]) / 2.5f, 3));
                            LevelLists.textures.Add(new float3(GetPlaneSignedDistanceToPoint(LeftPlane, LevelLists.vertices[baseVert + 1]) / 2.5f, GetPlaneSignedDistanceToPoint(TopPlane, LevelLists.vertices[baseVert + 1]) / 2.5f, 3));
                            LevelLists.textures.Add(new float3(GetPlaneSignedDistanceToPoint(LeftPlane, LevelLists.vertices[baseVert + 2]) / 2.5f, GetPlaneSignedDistanceToPoint(TopPlane, LevelLists.vertices[baseVert + 2]) / 2.5f, 3));
                            LevelLists.textures.Add(new float3(GetPlaneSignedDistanceToPoint(LeftPlane, LevelLists.vertices[baseVert + 3]) / 2.5f, GetPlaneSignedDistanceToPoint(TopPlane, LevelLists.vertices[baseVert + 3]) / 2.5f, 3));

                            PolygonMeta transformedmesh = new PolygonMeta
                            {
                                plane = LevelLists.planes.Length,

                                collider = i,

                                opaque = i,

                                sectorId = i,

                                connectedSectorId = -1,

                                edgeStartIndex = -1,

                                edgeCount = -1,

                                triangleStartIndex = baseStartIndex,

                                triangleCount = 6
                            };

                            LevelLists.polygons.Add(transformedmesh);

                            MathematicalPlane plane = new MathematicalPlane
                            {
                                normal = n,
                                distance = -math.dot(n, v0)
                            };

                            LevelLists.planes.Add(plane);

                            polygonCount += 1;
                        }
                        else
                        {
                            double C0 = sector.ceilingHeight / 8 * 2.5f;
                            double C1 = sector.floorHeight / 8 * 2.5f;

                            int baseVert = LevelLists.vertices.Length;

                            int baseStartIndex = LevelLists.triangles.Length;

                            LevelLists.vertices.Add(new float3((float)Z1, (float)C1, (float)X1));
                            LevelLists.vertices.Add(new float3((float)Z1, (float)C0, (float)X1));
                            LevelLists.vertices.Add(new float3((float)Z0, (float)C0, (float)X0));
                            LevelLists.vertices.Add(new float3((float)Z0, (float)C1, (float)X0));

                            LevelLists.triangles.Add(baseVert);
                            LevelLists.triangles.Add(baseVert + 1);
                            LevelLists.triangles.Add(baseVert + 2);
                            LevelLists.triangles.Add(baseVert);
                            LevelLists.triangles.Add(baseVert + 2);
                            LevelLists.triangles.Add(baseVert + 3);

                            float3 v0 = LevelLists.vertices[baseVert];
                            float3 v1 = LevelLists.vertices[baseVert + 1];
                            float3 v2 = LevelLists.vertices[baseVert + 2];

                            float3 n = math.normalize(math.cross(v1 - v0, v2 - v0));

                            float3 leftPlaneNormal = math.normalize(v2 - v1);
                            float leftPlaneDistance = -math.dot(leftPlaneNormal, v1);

                            float3 topPlaneNormal = math.normalize(v1 - v0);
                            float topPlaneDistance = -math.dot(topPlaneNormal, v1);

                            LeftPlane = new MathematicalPlane { normal = leftPlaneNormal, distance = leftPlaneDistance };
                            TopPlane = new MathematicalPlane { normal = topPlaneNormal, distance = topPlaneDistance };

                            LevelLists.textures.Add(new float3(GetPlaneSignedDistanceToPoint(LeftPlane, LevelLists.vertices[baseVert]) / 2.5f, GetPlaneSignedDistanceToPoint(TopPlane, LevelLists.vertices[baseVert]) / 2.5f, 3));
                            LevelLists.textures.Add(new float3(GetPlaneSignedDistanceToPoint(LeftPlane, LevelLists.vertices[baseVert + 1]) / 2.5f, GetPlaneSignedDistanceToPoint(TopPlane, LevelLists.vertices[baseVert + 1]) / 2.5f, 3));
                            LevelLists.textures.Add(new float3(GetPlaneSignedDistanceToPoint(LeftPlane, LevelLists.vertices[baseVert + 2]) / 2.5f, GetPlaneSignedDistanceToPoint(TopPlane, LevelLists.vertices[baseVert + 2]) / 2.5f, 3));
                            LevelLists.textures.Add(new float3(GetPlaneSignedDistanceToPoint(LeftPlane, LevelLists.vertices[baseVert + 3]) / 2.5f, GetPlaneSignedDistanceToPoint(TopPlane, LevelLists.vertices[baseVert + 3]) / 2.5f, 3));

                            PolygonMeta transformedmesh = new PolygonMeta
                            {
                                plane = LevelLists.planes.Length,

                                collider = i,

                                opaque = i,

                                sectorId = i,

                                connectedSectorId = -1,

                                edgeStartIndex = -1,

                                edgeCount = -1,

                                triangleStartIndex = baseStartIndex,

                                triangleCount = 6
                            };

                            LevelLists.polygons.Add(transformedmesh);

                            MathematicalPlane plane = new MathematicalPlane
                            {
                                normal = n,
                                distance = -math.dot(n, v0)
                            };

                            LevelLists.planes.Add(plane);

                            polygonCount += 1;
                        }
                    }
                    if (sectors[wall].ceilingHeight != sectors[wall].floorHeight)
                    {
                        if (sector.ceilingHeight > sectors[wall].ceilingHeight)
                        {
                            Ceiling = sectors[wall].ceilingHeight / 8 * 2.5f;
                        }
                        else
                        {
                            Ceiling = sector.ceilingHeight / 8 * 2.5f;
                        }
                        if (sector.floorHeight > sectors[wall].floorHeight)
                        {
                            Floor = sector.floorHeight / 8 * 2.5f;
                        }
                        else
                        {
                            Floor = sectors[wall].floorHeight / 8 * 2.5f;
                        }

                        int baseVert = LevelLists.vertices.Length;

                        int baseStartIndex = LevelLists.edges.Length;

                        LevelLists.vertices.Add(new float3((float)Z1, (float)Floor, (float)X1));
                        LevelLists.vertices.Add(new float3((float)Z1, (float)Ceiling, (float)X1));
                        LevelLists.vertices.Add(new float3((float)Z0, (float)Ceiling, (float)X0));
                        LevelLists.vertices.Add(new float3((float)Z0, (float)Floor, (float)X0));

                        LevelLists.edges.Add(baseVert);
                        LevelLists.edges.Add(baseVert + 1);
                        LevelLists.edges.Add(baseVert + 1);
                        LevelLists.edges.Add(baseVert + 2);
                        LevelLists.edges.Add(baseVert + 2);
                        LevelLists.edges.Add(baseVert + 3);
                        LevelLists.edges.Add(baseVert + 3);
                        LevelLists.edges.Add(baseVert);

                        float3 v0 = LevelLists.vertices[baseVert];
                        float3 v1 = LevelLists.vertices[baseVert + 1];
                        float3 v2 = LevelLists.vertices[baseVert + 2];

                        float3 n = math.normalize(math.cross(v1 - v0, v2 - v0));

                        LevelLists.textures.Add(float3.zero);
                        LevelLists.textures.Add(float3.zero);
                        LevelLists.textures.Add(float3.zero);
                        LevelLists.textures.Add(float3.zero);

                        PolygonMeta transformedmesh = new PolygonMeta
                        {
                            plane = LevelLists.planes.Length,

                            collider = -1,

                            opaque = -1,

                            sectorId = i,

                            connectedSectorId = wall,

                            edgeStartIndex = baseStartIndex,

                            edgeCount = 8,

                            triangleStartIndex = -1,

                            triangleCount = -1
                        };

                        LevelLists.polygons.Add(transformedmesh);

                        MathematicalPlane plane = new MathematicalPlane
                        {
                            normal = n,
                            distance = -math.dot(n, v0)
                        };

                        LevelLists.planes.Add(plane);

                        polygonCount += 1;
                    }

                    if (sector.floorHeight < sectors[wall].floorHeight)
                    {
                        if (sector.ceilingHeight > sectors[wall].floorHeight)
                        {
                            double F0 = sector.floorHeight / 8 * 2.5f;

                            if (sector.floorHeight > sectors[wall].floorHeight)
                            {
                                Floor = sector.floorHeight / 8 * 2.5f;
                            }
                            else
                            {
                                Floor = sectors[wall].floorHeight / 8 * 2.5f;
                            }

                            int baseVert = LevelLists.vertices.Length;

                            int baseStartIndex = LevelLists.triangles.Length;

                            LevelLists.vertices.Add(new float3((float)Z1, (float)F0, (float)X1));
                            LevelLists.vertices.Add(new float3((float)Z1, (float)Floor, (float)X1));
                            LevelLists.vertices.Add(new float3((float)Z0, (float)Floor, (float)X0));
                            LevelLists.vertices.Add(new float3((float)Z0, (float)F0, (float)X0));

                            LevelLists.triangles.Add(baseVert);
                            LevelLists.triangles.Add(baseVert + 1);
                            LevelLists.triangles.Add(baseVert + 2);
                            LevelLists.triangles.Add(baseVert);
                            LevelLists.triangles.Add(baseVert + 2);
                            LevelLists.triangles.Add(baseVert + 3);

                            float3 v0 = LevelLists.vertices[baseVert];
                            float3 v1 = LevelLists.vertices[baseVert + 1];
                            float3 v2 = LevelLists.vertices[baseVert + 2];

                            float3 n = math.normalize(math.cross(v1 - v0, v2 - v0));

                            float3 leftPlaneNormal = math.normalize(v2 - v1);
                            float leftPlaneDistance = -math.dot(leftPlaneNormal, v1);

                            float3 topPlaneNormal = math.normalize(v1 - v0);
                            float topPlaneDistance = -math.dot(topPlaneNormal, v1);

                            LeftPlane = new MathematicalPlane { normal = leftPlaneNormal, distance = leftPlaneDistance };
                            TopPlane = new MathematicalPlane { normal = topPlaneNormal, distance = topPlaneDistance };

                            LevelLists.textures.Add(new float3(GetPlaneSignedDistanceToPoint(LeftPlane, LevelLists.vertices[baseVert]) / 2.5f, GetPlaneSignedDistanceToPoint(TopPlane, LevelLists.vertices[baseVert]) / 2.5f, 2));
                            LevelLists.textures.Add(new float3(GetPlaneSignedDistanceToPoint(LeftPlane, LevelLists.vertices[baseVert + 1]) / 2.5f, GetPlaneSignedDistanceToPoint(TopPlane, LevelLists.vertices[baseVert + 1]) / 2.5f, 2));
                            LevelLists.textures.Add(new float3(GetPlaneSignedDistanceToPoint(LeftPlane, LevelLists.vertices[baseVert + 2]) / 2.5f, GetPlaneSignedDistanceToPoint(TopPlane, LevelLists.vertices[baseVert + 2]) / 2.5f, 2));
                            LevelLists.textures.Add(new float3(GetPlaneSignedDistanceToPoint(LeftPlane, LevelLists.vertices[baseVert + 3]) / 2.5f, GetPlaneSignedDistanceToPoint(TopPlane, LevelLists.vertices[baseVert + 3]) / 2.5f, 2));

                            PolygonMeta transformedmesh = new PolygonMeta
                            {
                                plane = LevelLists.planes.Length,

                                collider = i,

                                opaque = i,

                                sectorId = i,

                                connectedSectorId = -1,

                                edgeStartIndex = -1,

                                edgeCount = -1,

                                triangleStartIndex = baseStartIndex,

                                triangleCount = 6
                            };

                            LevelLists.polygons.Add(transformedmesh);

                            MathematicalPlane plane = new MathematicalPlane
                            {
                                normal = n,
                                distance = -math.dot(n, v0)
                            };

                            LevelLists.planes.Add(plane);

                            polygonCount += 1;
                        }
                        else
                        {
                            double F0 = sector.floorHeight / 8 * 2.5f;
                            double F1 = sector.ceilingHeight / 8 * 2.5f;

                            int baseVert = LevelLists.vertices.Length;

                            int baseStartIndex = LevelLists.triangles.Length;

                            LevelLists.vertices.Add(new float3((float)Z1, (float)F0, (float)X1));
                            LevelLists.vertices.Add(new float3((float)Z1, (float)F1, (float)X1));
                            LevelLists.vertices.Add(new float3((float)Z0, (float)F1, (float)X0));
                            LevelLists.vertices.Add(new float3((float)Z0, (float)F0, (float)X0));

                            LevelLists.triangles.Add(baseVert);
                            LevelLists.triangles.Add(baseVert + 1);
                            LevelLists.triangles.Add(baseVert + 2);
                            LevelLists.triangles.Add(baseVert);
                            LevelLists.triangles.Add(baseVert + 2);
                            LevelLists.triangles.Add(baseVert + 3);

                            float3 v0 = LevelLists.vertices[baseVert];
                            float3 v1 = LevelLists.vertices[baseVert + 1];
                            float3 v2 = LevelLists.vertices[baseVert + 2];

                            float3 n = math.normalize(math.cross(v1 - v0, v2 - v0));

                            float3 leftPlaneNormal = math.normalize(v2 - v1);
                            float leftPlaneDistance = -math.dot(leftPlaneNormal, v1);

                            float3 topPlaneNormal = math.normalize(v1 - v0);
                            float topPlaneDistance = -math.dot(topPlaneNormal, v1);

                            LeftPlane = new MathematicalPlane { normal = leftPlaneNormal, distance = leftPlaneDistance };
                            TopPlane = new MathematicalPlane { normal = topPlaneNormal, distance = topPlaneDistance };

                            LevelLists.textures.Add(new float3(GetPlaneSignedDistanceToPoint(LeftPlane, LevelLists.vertices[baseVert]) / 2.5f, GetPlaneSignedDistanceToPoint(TopPlane, LevelLists.vertices[baseVert]) / 2.5f, 2));
                            LevelLists.textures.Add(new float3(GetPlaneSignedDistanceToPoint(LeftPlane, LevelLists.vertices[baseVert + 1]) / 2.5f, GetPlaneSignedDistanceToPoint(TopPlane, LevelLists.vertices[baseVert + 1]) / 2.5f, 2));
                            LevelLists.textures.Add(new float3(GetPlaneSignedDistanceToPoint(LeftPlane, LevelLists.vertices[baseVert + 2]) / 2.5f, GetPlaneSignedDistanceToPoint(TopPlane, LevelLists.vertices[baseVert + 2]) / 2.5f, 2));
                            LevelLists.textures.Add(new float3(GetPlaneSignedDistanceToPoint(LeftPlane, LevelLists.vertices[baseVert + 3]) / 2.5f, GetPlaneSignedDistanceToPoint(TopPlane, LevelLists.vertices[baseVert + 3]) / 2.5f, 2));

                            PolygonMeta transformedmesh = new PolygonMeta
                            {
                                plane = LevelLists.planes.Length,

                                collider = i,

                                opaque = i,

                                sectorId = i,

                                connectedSectorId = -1,

                                edgeStartIndex = -1,

                                edgeCount = -1,

                                triangleStartIndex = baseStartIndex,

                                triangleCount = 6
                            };

                            LevelLists.polygons.Add(transformedmesh);

                            MathematicalPlane plane = new MathematicalPlane
                            {
                                normal = n,
                                distance = -math.dot(n, v0)
                            };

                            LevelLists.planes.Add(plane);

                            polygonCount += 1;
                        }
                    }
                }
            }

            if (sector.floorHeight != sector.ceilingHeight)
            {
                floorverts.Clear();
                ceilingverts.Clear();
                flooruvs.Clear();
                ceilinguvs.Clear();

                float tinyNumber = 1e-6f;

                for (int e = 0; e < sector.vertexIndices.Count; ++e)
                {
                    double YF = sector.floorHeight / 8 * 2.5f;
                    double YC = sector.ceilingHeight / 8 * 2.5f;
                    double X = vertices[sector.vertexIndices[e]].x / 2 * 2.5f;
                    double Z = vertices[sector.vertexIndices[e]].y / 2 * 2.5f;

                    float OX = (float)X / 2.5f * -1;
                    float OY = (float)Z / 2.5f;

                    floorverts.Add(new Vector3((float)Z, (float)YF, (float)X));
                    ceilingverts.Add(new Vector3((float)Z, (float)YC, (float)X));
                    flooruvs.Add(new Vector3(OY, OX, 0));
                    ceilinguvs.Add(new Vector3(OY, OX, 1));
                }

                floortri.Clear();

                for (int e = 0; e < floorverts.Count - 2; e++)
                {
                    Vector3 v0 = floorverts[0];
                    Vector3 v1 = floorverts[e + 1];
                    Vector3 v2 = floorverts[e + 2];

                    Vector3 e0 = v1 - v0;
                    Vector3 e1 = v2 - v1;
                    Vector3 e2 = v2 - v0;

                    if (e0.sqrMagnitude < tinyNumber || e1.sqrMagnitude < tinyNumber || e2.sqrMagnitude < tinyNumber)
                    {
                        continue;
                    }

                    Vector3 edges = Vector3.Cross(e0, e2);

                    if (edges.sqrMagnitude < tinyNumber)
                    {
                        continue;
                    }

                    floortri.Add(0);
                    floortri.Add(e + 1);
                    floortri.Add(e + 2);
                }

                ceilingverts.Reverse();
                ceilinguvs.Reverse();

                ceilingtri.Clear();

                for (int e = 0; e < ceilingverts.Count - 2; e++)
                {
                    Vector3 v0 = ceilingverts[0];
                    Vector3 v1 = ceilingverts[e + 1];
                    Vector3 v2 = ceilingverts[e + 2];

                    Vector3 e0 = v1 - v0;
                    Vector3 e1 = v2 - v1;
                    Vector3 e2 = v2 - v0;

                    if (e0.sqrMagnitude < tinyNumber || e1.sqrMagnitude < tinyNumber || e2.sqrMagnitude < tinyNumber)
                    {
                        continue;
                    }

                    Vector3 edges = Vector3.Cross(e0, e2);

                    if (edges.sqrMagnitude < tinyNumber)
                    {
                        continue;
                    }

                    ceilingtri.Add(0);
                    ceilingtri.Add(e + 1);
                    ceilingtri.Add(e + 2);
                }

                int baseFloor = LevelLists.vertices.Length;

                int floorStartIndex = LevelLists.triangles.Length;

                for (int e = 0; e < floorverts.Count; e++)
                {
                    LevelLists.vertices.Add(floorverts[e]);
                }

                for (int e = 0; e < flooruvs.Count; e++)
                {
                    LevelLists.textures.Add(flooruvs[e]);
                }

                for (int e = 0; e < floortri.Count; e++)
                {
                    LevelLists.triangles.Add(baseFloor + floortri[e]);
                }

                float3 f0 = floorverts[floortri[0]];
                float3 f1 = floorverts[floortri[1]];
                float3 f2 = floorverts[floortri[2]];

                float3 f = math.normalize(math.cross(f1 - f0, f2 - f0));

                PolygonMeta transformedfloormesh = new PolygonMeta
                {
                    plane = LevelLists.planes.Length,

                    collider = i,

                    opaque = i,

                    sectorId = i,

                    connectedSectorId = -1,

                    edgeStartIndex = -1,

                    edgeCount = -1,

                    triangleStartIndex = floorStartIndex,

                    triangleCount = floortri.Count
                };

                LevelLists.polygons.Add(transformedfloormesh);

                MathematicalPlane floorPlane = new MathematicalPlane
                {
                    normal = f,
                    distance = -math.dot(f, f0)
                };

                LevelLists.planes.Add(floorPlane);

                polygonCount += 1;

                int baseCeiling = LevelLists.vertices.Length;

                int ceilingStartIndex = LevelLists.triangles.Length;

                for (int e = 0; e < ceilingverts.Count; e++)
                {
                    LevelLists.vertices.Add(ceilingverts[e]);
                }

                for (int e = 0; e < ceilinguvs.Count; e++)
                {
                    LevelLists.textures.Add(ceilinguvs[e]);
                }

                for (int e = 0; e < ceilingtri.Count; e++)
                {
                    LevelLists.triangles.Add(baseCeiling + ceilingtri[e]);
                }

                float3 c0 = ceilingverts[ceilingtri[0]];
                float3 c1 = ceilingverts[ceilingtri[1]];
                float3 c2 = ceilingverts[ceilingtri[2]];

                float3 c = math.normalize(math.cross(c1 - c0, c2 - c0));

                PolygonMeta transformedceilingmesh = new PolygonMeta
                {
                    plane = LevelLists.planes.Length,

                    collider = i,

                    opaque = i,

                    sectorId = i,

                    connectedSectorId = -1,

                    edgeStartIndex = -1,

                    edgeCount = -1,

                    triangleStartIndex = ceilingStartIndex,

                    triangleCount = ceilingtri.Count
                };

                LevelLists.polygons.Add(transformedceilingmesh);

                MathematicalPlane ceilingPlane = new MathematicalPlane
                {
                    normal = c,
                    distance = -math.dot(c, c0)
                };

                LevelLists.planes.Add(ceilingPlane);

                polygonCount += 1;
            }

            SectorMeta sectorMeta = new SectorMeta
            {
                sectorId = i,
                polygonStartIndex = polygonStart,
                polygonCount = polygonCount,
                planeStartIndex = 0,
                planeCount = 4
            };

            LevelLists.sectors.Add(sectorMeta);
            polygonStart += polygonCount;
        }

        Debug.Log("Level built successfully!");
    }

    public void BuildObjects()
    {
        for (int i = 0; i < starts.Count; i++)
        {
            StartPosition start = new StartPosition
            {
                playerStart = new Vector3(starts[i].location.x / 2 * 2.5f, sectors[starts[i].sector].floorHeight / 8 * 2.5f, starts[i].location.y / 2 * 2.5f),

                sectorId = starts[i].sector
            };

            LevelLists.positions.Add(start);
        }
    }

    public void BuildColliders()
    {
        for (int i = 0; i < LevelLists.sectors.Length; i++)
        {
            OpaqueVertices.Clear();

            OpaqueTriangles.Clear();

            int triangleCount = 0;

            for (int e = LevelLists.sectors[i].polygonStartIndex; e < LevelLists.sectors[i].polygonStartIndex + LevelLists.sectors[i].polygonCount; e++)
            {
                if (LevelLists.polygons[e].collider != -1)
                {
                    for (int f = LevelLists.polygons[e].triangleStartIndex; f < LevelLists.polygons[e].triangleStartIndex + LevelLists.polygons[e].triangleCount; f += 3)
                    {
                        OpaqueVertices.Add(LevelLists.vertices[LevelLists.triangles[f]]);
                        OpaqueVertices.Add(LevelLists.vertices[LevelLists.triangles[f + 1]]);
                        OpaqueVertices.Add(LevelLists.vertices[LevelLists.triangles[f + 2]]);
                        OpaqueTriangles.Add(triangleCount);
                        OpaqueTriangles.Add(triangleCount + 1);
                        OpaqueTriangles.Add(triangleCount + 2);
                        triangleCount += 3;
                    }
                } 
            }

            Mesh combinedmesh = new Mesh();

            CollisionMesh.Add(combinedmesh);

            combinedmesh.SetVertices(OpaqueVertices);

            combinedmesh.SetTriangles(OpaqueTriangles, 0);

            GameObject meshObject = new GameObject("Collision " + i);

            MeshCollider meshCollider = meshObject.AddComponent<MeshCollider>();

            meshCollider.sharedMesh = combinedmesh;

            CollisionSectors.Add(meshCollider);

            meshObject.transform.SetParent(CollisionObjects.transform);
        }
    }
}
