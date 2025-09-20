# LEASE BNB

# Step 1 Collect Data : PostgreSQL every Field is important for RAG

    @Entity("properties")
    export class Property extends BaseEntity {
    @PrimaryGeneratedColumn("uuid")
    id: string;
    @Column({ type: "varchar", length: 255, nullable: true })
    property_title: string;
    @Column({ type: "text", nullable: true })
    property_description: string;
    @ManyToOne(() => PropertyType, { nullable: true })
    @JoinColumn({ name: "property_type_id" })
    property_type: PropertyType;
    @Column({
    type: "enum",
    enum: FurnishingStatus,
    default: FurnishingStatus.UNFURNISHED,
    nullable: true,
    })
    furnishing_status: FurnishingStatus;
    @Column({ type: "float", nullable: true })
    number_of_bathrooms: number;
    @Column({ type: "float", nullable: true })
    number_of_bedrooms: number;
    @Column({ type: "float", nullable: true })
    property_size: number;
    @Column({ type: "int", nullable: true })
    year_built: number;
    @Column({ type: "varchar", nullable: true })
    floor_level: string;
    @Column({ type: "varchar", length: 255, nullable: true })
    emirate: string;
    @Column({ type: "varchar", length: 255, nullable: true })
    city: string;
    @Column({ type: "varchar", length: 255, nullable: true })
    community: string;
    @Column({ type: "varchar", length: 255, nullable: true })
    subcommunity: string;
    @Column({ type: "varchar", length: 255, nullable: true })
    building_name: string;
    @Column({ type: "varchar", length: 255, nullable: true })
    apartment_unit_number: string;
    @Column({ type: "text", nullable: true })
    nearby_landmarks: string;
    @Column({ type: "int", nullable: true })
    plot_number: number;
    @Column({ type: "varchar", length: 255, nullable: true })
    developer_name: string;
    @ManyToOne(() => PropertyRentType, { nullable: true })
    @JoinColumn({ name: "rent_type_id" }) // Lease,Leasing - Holiday Home Ready,Management Fees
    rent_type: PropertyRentType;
    @Column({ type: "float", nullable: true })
    rent_charge: number;
    @Column({ type: "varchar", length: 255, nullable: true })
    lease_duration: string;
    @Column({ type: "varchar", length: 255, nullable: true })
    payment_terms: string;
    @Column({ type: "boolean", nullable: true })
    sublease_allowed: boolean;
    @Column({ type: "timestamptz", nullable: true })
    lease_start_date: Date;
    @Column({ type: "timestamptz", nullable: true })
    lease_end_date: Date;
    @Column({ type: "float", nullable: true })
    security_deposit: number;
    @Column({ type: "varchar", length: 255, nullable: true })
    setup_grace_period: string;
    @Column({ type: "float", nullable: true })
    maintenance_charge: number;
    @Column({ type: "enum", enum: MaintenanceCoveredBy, nullable: true })
    maintenance_covered_by: MaintenanceCoveredBy;
    @Column({ type: "json", nullable: true })
    parking: any;
    // "parking": [
    // {
    // number_of_slots: 4;
    // type: "Open Parking";
    // },
    // {
    // number_of_slots: 4;
    // type: "Closed Parking";
    // }
    // ];
    @Column({ type: "json", nullable: true })
    swimming_pool: any;
    // "swimming_pool": [
    // {
    // number_of_pools: 3;
    // type: "Shared Pool";
    // },
    // {
    // number_of_pools: 3;
    // type: "Private Pool";
    // }
    // ];
    @Column({ type: "boolean", default: false })
    maids_room: boolean;
    @Column({ type: "boolean", default: false })
    security_available: boolean;
    @Column({ type: "boolean", default: false })
    concierge_available: boolean;
    @Column({ type: "boolean", default: false })
    central_ac_heating: boolean;
    @Column({ type: "boolean", default: false })
    elevators: boolean;
    @Column({ type: "boolean", default: false })
    balcony_terrace: boolean;
    @Column({ type: "boolean", default: false })
    storage_room: boolean;
    @Column({ type: "boolean", default: false })
    laundry_room: boolean;
    @Column({ type: "boolean", default: false })
    gym_fitness_center: boolean;
    @Column({ type: "boolean", default: false })
    childrens_play_area: boolean;
    @Column({ type: "boolean", default: false })
    bbq_area: boolean;
    @Column({ type: "boolean", default: false })
    pet_friendly: boolean;
    @Column({ type: "boolean", default: false })
    smart_home_features: boolean; //discussion
    @Column({ type: "boolean", default: false })
    waste_disposal_system: boolean;
    @Column({ type: "boolean", default: false })
    power_backup: boolean;
    @Column({ type: "boolean", default: false })
    mosque_nearby: boolean;
    @Column({ type: "boolean", default: false })
    beach_access: boolean;
    @Column({ type: "boolean", default: false })
    jogging_cycling_tracks: boolean;
    @Column({ type: "uuid", nullable: true })
    property_view_id: string;
    @Column({ type: "int", nullable: true })
    unit_number: number;
    @Column({ type: "json", nullable: true })
    public_transport_type: string[]; // Storing array of strings instead of enum
    @Column({ type: "json", nullable: true })
    retail_shopping_access: string[]; // Storing array of strings instead of relation
    @Column({ type: "timestamptz", nullable: true })
    available_from: Date;
    @Column({ type: "timestamptz", nullable: true })
    tenancy_start_date: Date;
    @Column({ type: "timestamptz", default: () => "CURRENT_TIMESTAMP" })
    listing_date: Date;
    @Column({ type: "boolean", default: false })
    chiller_included: boolean;
    @Column({
    type: "enum",
    enum: PropertyStatus,
    default: PropertyStatus.DRAFT,
    nullable: true,
    })
    property_status: PropertyStatus;
    @Column({
    type: "enum",
    enum: BoostingStatus,
    default: BoostingStatus.NOT_BOOSTED,
    })
    premiumBoostingStatus: BoostingStatus; // IMP: Property which are Premium
    @Column({
    type: "enum",
    enum: BoostingStatus,
    default: BoostingStatus.NOT_BOOSTED,
    })
    carouselBoostingStatus: BoostingStatus; // IMP: Property which are Prime
    @Column({
    type: "enum",
    enum: BnbVerificationStatus,
    nullable: true,
    default: BnbVerificationStatus.NOT_REQUIRED,
    })
    bnb_verification_status: BnbVerificationStatus; // IMP: Property which are Verified by admin
    @OneToMany(() => PropertyMedia, (media) => media.property, {
    cascade: true,
    })
    media: PropertyMedia[];
    }
    enum FurnishingStatus {
    FURNISHED = "furnished",
    SEMI_FURNISHED = "semi-furnished",
    UNFURNISHED = "unfurnished",
    }
    enum MaintenanceCoveredBy {
    OWNER = "owner",
    TENANT = "tenant",
    SHARED = "shared",
    }
    enum BnbVerificationStatus {
    NOT_REQUIRED = "notrequired",
    REQUESTED = "requested",
    VERIFIED = "verified",
    IN_PROGRESS = "inprogress",
    REJECTED = "rejected",
    FAILED = "failed",
    }
    enum BoostingStatus {
    NOT_BOOSTED = "NotBoosted",
    ACTIVE = "Active",
    EXPIRED = "Expired",
    CANCELLED = "Cancelled",
    REQUESTED = "Requested",
    }
    enum PropertyStatus {
    DRAFT = "draft",
    REVIEW = "review",
    REVISE = "revise",
    LISTED = "listed",
    UNLISTED = "unlisted",
    HOLD = "hold",
    SUSPENDED = "suspended",
    EXPIRED = "expired",
    OFF_MARKET = "off_market",
    ACTIVE = "active",
    }

# Step 2 Chunk Data : Split into small overlapping text chunks. Chunking and Embedding should happen only for property_status which is listed.

# Step 3 Embed Chunks : Convert chunks into vector embeddings. Please suggest me which field be used for semantic and for filter

# Step 4 Store Embeddings : Save them in a vector database (elastic (8.17.1) with knn and with index name rag_properties) For Embedding use google gemini (gemini-embedding-001)

# Step 5 User Query : Categories of Questions Users Can Ask

    Below, I've expanded on the 10 categories from the previous response. For each category, I've provided at least 5 sample questions that a user might ask in a RAG-based AI chat for leasing properties. These are tailored to the Property entity attributes, enums, and relationships, assuming the AI can filter and retrieve relevant listings.

    1. Location-Based Queries
    "Find me apartments for lease in Dubai Marina, Dubai."
    "What properties are available in Jumeirah Village Circle, JVC?"
    "Show villas near Sheikh Zayed Road in Dubai."
    "List houses for rent in Al Raha Beach, Abu Dhabi."
    "Are there any condos in Dubai Hills Estate with nearby landmarks like malls?"
    "Find properties in Bur Dubai, close to the metro station."
    2. Property Type and Characteristics
    "Show me 1-bedroom apartments for lease in Dubai."
    "Find villas with 4 or more bedrooms in Abu Dhabi."
    "What townhouses are available with at least 2 bathrooms?"
    "List properties larger than 2,000 sq.ft. built before 2010."
    "Are there duplexes on the 5th floor or higher in Business Bay?"
    "Find me a studio apartment with a property size under 800 sq.ft. in JLT."
    3. Furnishing and Amenities
    "Find semi-furnished 2-bedroom properties in Dubai Marina."
    "Show apartments with a shared swimming pool and 3 parking slots."
    "What villas have a gym, balcony, and central AC?"
    "List unfurnished houses with a maid's room and storage."
    "Are there pet-friendly condos with a BBQ area and elevators?"
    "Find properties with smart home features and a laundry room."
    4. Lease Terms and Financials
    "Find properties under AED 150,000/year rent in Dubai."
    "Show 24-month leases with monthly payment terms."
    "What apartments have maintenance covered by the tenant?"
    "List villas with a security deposit of AED 5,000 or less."
    "Are there properties allowing sublease with no setup grace period?"
    "Find leases starting January 1, 2026, with a rent charge around AED 200,000."
    5. Availability and Move-In Dates
    "Show properties available from December 2025 in Dubai."
    "Find apartments ready to move in by mid-November 2025."
    "What villas can I lease starting immediately in Abu Dhabi?"
    "List houses with tenancy start dates in early 2026."
    "Are there condos available after October 31, 2025, in JBR?"
    "Find properties listed within the last 30 days for quick move-in."
    6. Property Status and Verification
    "Show active and listed properties for lease in Dubai."
    "Find premium-boosted apartments in Downtown Dubai."
    "What verified properties (bnb_verification_status: VERIFIED) are available?"
    "List unlisted or hold status properties in Abu Dhabi."
    "Are there expired boosting status villas in Palm Jumeirah?"
    "Find properties in review or revise status for potential leasing."
    7. Additional Features and Accessibility
    "Find apartments near metro and bus public transport in Dubai Marina."
    "Show villas with direct beach access in Jumeirah."
    "What properties have jogging tracks and a mosque nearby?"
    "List condos with easy retail shopping access like supermarkets."
    "Are there pet-friendly houses with cycling tracks in Al Barsha?"
    "Find properties with power backup and waste disposal systems close to schools."
    8. Developer and Property Details
    "Show Emaar-developed apartments for lease in Dubai."
    "Find properties with plot number 123 in Emirates Hills."
    "What villas have apartment unit number 456?"
    "List Nakheel buildings available in Palm Jumeirah."
    "Are there Aldar properties with unit number 789 in Abu Dhabi?"
    "Find developer-specific houses like Sobha Realty in Dubai Hills."
    9. Combination Queries
    "Find a 3-bedroom furnished villa in Palm Jumeirah, budget AED 250,000/year, with private pool, 12-month lease, available by December 2025."
    "Show 1-bedroom apartments in Dubai Marina, semi-furnished, under AED 100,000/year, pet-friendly, with gym and beach access, verified status."
    "List 2-bedroom houses in Abu Dhabi, Al Reem Island, unfurnished, 6-month lease, maintenance by owner, balcony, near public transport."
    "Find a penthouse in Business Bay, 4 bathrooms, built after 2020, budget AED 180,000/year, with parking and concierge, ready by November 1, 2025."
    "What 4-bedroom villas in Emirates Hills, fully furnished, AED 300,000/year, sublease allowed, with maid's room and jogging tracks, active listing."
    "Show townhouses in JLT, 2 bedrooms, AED 120,000/year, 24-month lease, central AC, smart home features, no grace period, with media videos."

# Step 6 Embed Query : Convert the query into an embedding(gemini-embedding-001)

# Step 7 Retrieve Context : Find top-k similar chunks from the vector database.

    Prioritizing Context Retrieval for LLM in a Vector Database Search
    When retrieving context for a user query like "Find a 2-bedroom furnished villa in Dubai Hills, budget AED 150,000/year, with a private pool and 12-month lease, available by December 1, 2025" in a RAG-based AI chat system integrated with a vector database, the LLM must prioritize the most relevant attributes to find the top-k similar chunks (representing property listings) from the Property entity schema. This is critical when no exact matches are found (e.g., no properties in community: Dubai Hills or emirate: Dubai), as the LLM needs to suggest alternates and gather user requirements, as specified in your requirements. The vector database stores embeddings of property data (e.g., attributes like emirate, city, community, rent_charge, number_of_bedrooms, furnishing_status, property_type, swimming_pool, lease_duration, available_from), and the LLM retrieves chunks based on semantic similarity to the query.
    The challenge is to determine which attributes the LLM should prioritize when computing similarity and retrieving the top-k chunks, especially when the primary location (e.g., Dubai Hills) is unavailable. This response outlines the prioritization strategy for context retrieval, explains why certain attributes (e.g., price, bedrooms) are less useful without a viable location, and integrates the requirement to offer alternate questions and gather user requirements when no matches are found.
    Prioritization Strategy for Context Retrieval
    The LLM prioritizes attributes based on their influence on user intent and their impact on semantic similarity in the vector database. The goal is to retrieve chunks that closely match the user’s preferences, focusing on what users typically prioritize in real estate searches (as outlined in the previous response). The prioritization also ensures that alternate suggestions are relevant when no exact matches are found, and requirements are gathered for the admin endpoint.
    Below is the prioritized list of attributes for context retrieval, with reasoning tied to user behavior and the Property entity schema. Each attribute is mapped to its role in the vector database (e.g., as part of the embedding or metadata filter). The priorities are ordered from highest to lowest, with guidance on handling no matches.
    Location (emirate, city, community, subcommunity, nearby_landmarks, public_transport_type)
    Priority: Highest
    Why Prioritize: Location is the most critical factor in real estate searches, as it determines proximity to work, schools, lifestyle hubs (e.g., Dubai Marina Mall), and transport (e.g., metro). Users often have a specific community (e.g., Dubai Hills) or emirate (e.g., Dubai) in mind due to reputation or accessibility. These fields are likely encoded in the vector embeddings (e.g., text descriptions like “Dubai Hills villa near golf course”) and used as metadata filters.
    Vector Database Role:
    Embeddings capture location-specific terms (e.g., “Dubai Hills,” “near Sheikh Zayed Road”).
    Metadata filters (e.g., emirate = 'Dubai', community = 'Dubai Hills') narrow down chunks.
    Retrieval Strategy:
    Prioritize chunks with exact or near-exact matches to community or emirate (e.g., cosine similarity on embeddings with “Dubai Hills”).
    If no matches, expand to nearby communities (e.g., Emirates Hills, Arabian Ranches) or adjacent emirates (e.g., Sharjah, Abu Dhabi) by relaxing the filter (e.g., emirate = 'Dubai' OR emirate = 'Sharjah').
    When No Matches:
    Alternate Question: Suggest nearby locations after verifying their existence (e.g., SELECT COUNT(\*) FROM properties WHERE community = 'Emirates Hills' or check vector database for chunks with “Emirates Hills”).
    Example: “No villas in Dubai Hills, but would you consider Emirates Hills or Arabian Ranches, 10-15 minutes away?”
    Context Check: Query the vector database for chunks with nearby locations (e.g., query: "villa in Emirates Hills") to ensure suggestions are viable.
    Budget (rent_charge, security_deposit, maintenance_charge)
    Priority: High
    Why Prioritize: Budget is a primary constraint, as users want affordable properties. The rent_charge (e.g., AED 150,000/year) is critical, with security_deposit and maintenance_charge as secondary cost factors. Users may stretch budgets by ±10-20% for the right property.
    Vector Database Role:
    Embeddings include price-related terms (e.g., “AED 150,000/year villa”).
    Metadata filters (e.g., rent_charge <= 150000) refine chunks.
    Retrieval Strategy:
    Weight chunks with rent_charge close to the user’s budget (e.g., within ±20%, AED 120,000–180,000).
    If no matches in the specified location, prioritize budget after location (e.g., find villas in Emirates Hills within AED 150,000).
    When No Matches:
    Alternate Question: Suggest a higher budget or lower-cost areas (e.g., JLT instead of Dubai Marina). Verify (e.g., query: "villa in Dubai rent_charge <= 180000").
    Example: “No villas under AED 150,000 in Dubai Hills, but would you consider up to AED 180,000 or JLT?”
    Why Less Useful Without Location: Adjusting budget alone (e.g., AED 180,000 in Dubai Hills) is ineffective if no properties exist in the location.
    Property Type (property_type)
    Priority: High
    Why Prioritize: The property type (e.g., villa, apartment) reflects lifestyle needs (e.g., villas for families). The PropertyType relation allows precise filtering. Users may accept similar types (e.g., townhouse) if the location is ideal.
    Vector Database Role:
    Embeddings encode type (e.g., “furnished villa”).
    Metadata filter (e.g., property_type_id = 'villa') narrows chunks.
    Retrieval Strategy:
    Prioritize chunks matching the exact property_type (e.g., villa).
    If no matches, include related types (e.g., townhouse) in the same location.
    When No Matches:
    Alternate Question: Suggest a related type in the same/nearby location. Verify (e.g., query: "townhouse in Dubai Hills").
    Example: “No villas in Dubai Hills, but would you consider a townhouse?”
    Number of Bedrooms/Bathrooms (number_of_bedrooms, number_of_bathrooms)
    Priority: Medium-High
    Why Prioritize: Bedrooms and bathrooms indicate space needs (e.g., 2 bedrooms for a small family). Users are often flexible by ±1 bedroom if other criteria (location, budget) are met.
    Vector Database Role:
    Embeddings include bedroom count (e.g., “2-bedroom villa”).
    Metadata filter (e.g., number_of_bedrooms = 2).
    Retrieval Strategy:
    Prioritize exact matches (e.g., 2 bedrooms).
    If no matches, include ±1 bedroom (e.g., 1 or 3 bedrooms) after location.
    When No Matches:
    Alternate Question: Suggest ±1 bedroom. Verify (e.g., query: "3-bedroom villa in Dubai").
    Example: “No 2-bedroom villas in Dubai Hills, but would you consider 3 bedrooms?”
    Why Less Useful Without Location: Adjusting bedrooms without a viable location (e.g., Dubai Hills) doesn’t yield results if no properties exist there.
    Furnishing Status (furnishing_status)
    Priority: Medium
    Why Prioritize: Furnishing (FURNISHED, SEMI_FURNISHED, UNFURNISHED) affects move-in ease. UAE expats often prefer furnished properties, but some are flexible to save costs.
    Vector Database Role:
    Embeddings include furnishing (e.g., “furnished villa”).
    Metadata filter (e.g., furnishing_status = 'FURNISHED').
    Retrieval Strategy:
    Prioritize exact matches (e.g., FURNISHED).
    If no matches, include SEMI_FURNISHED or UNFURNISHED.
    When No Matches:
    Alternate Question: Suggest alternate furnishing. Verify (e.g., query: "semi-furnished villa in Dubai").
    Example: “No furnished villas in Dubai Hills, but would you consider semi-furnished?”
    Key Amenities (swimming_pool, pet_friendly, gym_fitness_center, parking, balcony_terrace)
    Priority: Medium
    Why Prioritize: Amenities like a private pool or pet-friendliness are often deal-breakers. In the UAE, pools and parking are highly valued. Users prioritize 1-2 key amenities.
    Vector Database Role:
    Embeddings capture amenities (e.g., “villa with private pool”).
    Metadata filters (e.g., swimming_pool IS NOT NULL, pet_friendly = true).
    Retrieval Strategy:
    Prioritize chunks with key amenities (e.g., private pool).
    If no matches, relax secondary amenities (e.g., shared pool).
    When No Matches:
    Alternate Question: Suggest shared pool or other amenities. Verify (e.g., query: "villa with shared pool in Dubai").
    Example: “No villas with a private pool in Dubai Hills, but would you accept a shared pool?”
    Lease Duration (lease_duration, rent_type)
    Priority: Medium-Low
    Why Prioritize: Lease terms (e.g., 12 months) matter for flexibility, especially for short-term UAE residents. Users may adjust terms for the right property.
    Vector Database Role:
    Embeddings include lease terms (e.g., “12-month lease villa”).
    Metadata filter (e.g., lease_duration = '12 months').
    Retrieval Strategy:
    Prioritize exact lease duration.
    If no matches, include shorter/longer terms.
    When No Matches:
    Alternate Question: Suggest alternate terms. Verify (e.g., query: "6-month lease villa in Dubai").
    Example: “No 12-month lease villas in Dubai Hills, but would you consider 6 months?”
    Availability Date (available_from)
    Priority: Low
    Why Prioritize: Move-in dates are important but flexible (e.g., ±1 month). Users plan moves in advance (e.g., December 2025).
    Vector Database Role:
    Embeddings may include dates (e.g., “available December 2025”).
    Metadata filter (e.g., available_from <= '2025-12-01').
    Retrieval Strategy:
    Prioritize chunks within the requested date.
    If no matches, include ±1 month.
    When No Matches:
    Alternate Question: Suggest nearby dates. Verify (e.g., query: "villa available January 2026").
    Example: “No villas available by December 1, 2025, but would you consider January 2026?”
    Why This Prioritization for Vector Database Retrieval
    Location First: Users prioritize emirate/community due to lifestyle and logistics. Vector embeddings capture location-specific terms (e.g., “Dubai Hills”), and metadata filters ensure precise matches. Relaxing location to nearby areas (e.g., Emirates Hills) yields the most relevant alternates.
    Budget and Type Next: rent_charge and property_type are critical for affordability and suitability. These are weighted heavily in embeddings (e.g., “AED 150,000 villa”) but secondary to location, as price adjustments are ineffective without a viable area.
    Other Attributes: Bedrooms, furnishing, amenities, lease terms, and availability are progressively more flexible. They are encoded in embeddings and metadata but relaxed later to preserve user intent.
    Vector Similarity: Cosine similarity prioritizes chunks with matching location and key attributes (e.g., “2-bedroom furnished villa in Dubai Hills”). If no matches, the LLM retrieves chunks with high similarity to nearby locations or adjusted attributes.

# Step 8 Build Prompt : Combine retrieved chunks + the user’s question.

    I want Short Term Memory implementation By LLM
    As i have given the set of question what users can ask it may be different also they are just the example do not consider as permanent set of question. Question can be change or any question related to property
    I do not want just to create the ai chatbot which just answer the users query i want to create ai chatbot which understand users query and reply accordingly it should feel like he/she is talking to human which taking information or idea which property will be best for her/him based on users query. Ai chatbot and users should interactive conversation.
    Below is a sample interactive conversation between a user and a RAG-based AI chat system designed to search for properties for lease in the UAE, leveraging the provided Property entity schema. The conversation demonstrates how the AI can interpret user queries, filter properties using entity attributes (e.g., emirate, city, community, rent_charge, furnishing_status, number_of_bedrooms), handle vague or specific requests, and guide the user to refine their search. The AI uses natural language understanding to map user input to the schema and provides relevant responses, including follow-up questions to clarify preferences.
    Sample Interactive Conversation
    User: Hi, I want to lease a property in the UAE, but I’m not sure where to look. Can you help me find something?
    AI: I’d be happy to help you find a great property to lease in the UAE! To get started, could you share which city you’re considering, like Dubai or Abu Dhabi? Also, what’s your budget, and do you have any preferences for the number of bedrooms or whether the property is furnished?
    User: I’m leaning toward Dubai, maybe a nice area. I’m not familiar with the neighborhoods, though. My budget is around AED 100,000 per year, and I need at least a 1-bedroom place.
    AI: Dubai’s a great choice with plenty of vibrant areas! Since you’re new to the neighborhoods, I’d suggest popular spots like Dubai Marina, Downtown Dubai, or Jumeirah Beach Residence (JBR) for their lifestyle and amenities. For a 1-bedroom apartment within AED 100,000/year, here are some options in Dubai Marina:
    Cayan Tower: Furnished 1-bedroom, 800 sq.ft., with a shared pool, balcony, and central AC. Rent: AED 95,000/year. Available from November 15, 2025. Verified property.
    Marina Gate: Semi-furnished 1-bedroom, 750 sq.ft., with a pool and gym. Rent: AED 90,000/year. Available immediately.
    Al Sahab Tower 1: Unfurnished 1-bedroom, 700 sq.ft., near public transport. Rent: AED 85,000/year. Available from December 1, 2025.
    You can view the property card for these on the platform. Would you like to focus on Dubai Marina, or should I explore other areas like JLT or Downtown Dubai? Also, do you prefer a furnished apartment, and are there specific amenities you want, like a pool?
    User: Dubai Marina sounds perfect! I want a furnished apartment with a pool, and it needs to be available by December 2025.
    AI: Got it! Let’s narrow it down to furnished 1-bedroom apartments in Dubai Marina with a pool, available by December 2025, within your AED 100,000/year budget. Here are some top matches:
    Cayan Tower: Furnished 1-bedroom, 800 sq.ft., shared pool, central AC, balcony. Rent: AED 95,000/year. Available from November 15, 2025. Verified, premium boosted.
    The Address Dubai Marina: Furnished 1-bedroom, 850 sq.ft., pool, gym, concierge. Rent: AED 98,000/year. Available from December 1, 2025.
    Al Sahab Tower 2: Furnished 1-bedroom, 720 sq.ft., shared pool, elevators. Rent: AED 88,000/year. Available from December 1, 2025. Pet-friendly.
    These properties fit your criteria, and you can check their property cards for full details. Would you like more information on one of these, like lease terms or nearby landmarks? Or do you have additional preferences, like pet-friendliness or a specific lease duration?
    User: The Al Sahab Tower 2 sounds good since it’s cheaper. Can you tell me about the lease terms and if it’s pet-friendly? I have a small dog.
    AI: Nice choice with Al Sahab Tower 2 in Dubai Marina! Here’s the rundown for the furnished 1-bedroom apartment, including pet-friendliness and lease terms, which you can display on the property card:
    Property Details: Furnished 1-bedroom, 720 sq.ft., AED 88,000/year, shared pool, elevators, central AC. Available from December 1, 2025.
    Pet-Friendly: Yes, suitable for a small dog (pet_friendly: true).
    Lease Terms:
    Rent Charge: AED 88,000/year.
    Lease Duration: 12 months (renewable).
    Security Deposit: AED 4,500 (refundable).
    Maintenance Covered By: Owner (no extra maintenance charge).
    Setup Grace Period: None (move-in ready by December 1, 2025).
    Sublease Allowed: Yes.
    Verification Status: Verified (bnb_verification_status: VERIFIED).
    This property is a great fit for you and your dog. Would you like details on nearby landmarks or public transport to add to the property card? Or should I check for other pet-friendly options in Dubai Marina with similar features?
    User: That’s great! Can you tell me about nearby landmarks and public transport for Al Sahab Tower 2?
    AI: Absolutely, here’s the information for Al Sahab Tower 2 in Dubai Marina regarding nearby landmarks and public transport, perfect for including in the property card:
    Property Recap: Furnished 1-bedroom, 720 sq.ft., AED 88,000/year, pet-friendly, shared pool, available from December 1, 2025.
    Nearby Landmarks (nearby_landmarks):
    Dubai Marina Mall: 5-minute walk (500 meters), with shops, restaurants, and a cinema.
    Marina Walk: Adjacent to the building, offering waterfront dining and cafes.
    Jumeirah Beach: 10-minute walk (1 km), ideal for beach outings.
    The Walk at JBR: 15-minute walk (1.5 km), a bustling area for shopping and dining.
    Public Transport (public_transport_type):
    DMCC Metro Station: 7-minute walk (700 meters), Dubai Metro Red Line, connecting to Downtown Dubai and beyond.
    Bus Stops: Within 5 minutes, with routes to Sheikh Zayed Road.
    Tram: Dubai Marina Tram Station, 10-minute walk, linking to JBR and Al Sufouh.
    This location is super convenient for both lifestyle and commuting. Would you like me to include specific lease payment terms (e.g., monthly or quarterly) for the property card, or are you ready to move forward with this one?
    User: This sounds perfect. Can you confirm the lease can start in early December, and how do I proceed?
    AI: I’m glad you’re happy with the Al Sahab Tower 2 apartment! I can confirm the lease is flexible to start from December 1, 2025 (lease_start_date), which fits your early December timeline. Here’s how you can proceed to lease this property, and you can include these steps in the property card:
    Property Recap for Card:
    Location: Al Sahab Tower 2, Dubai Marina, Dubai, UAE.
    Details: Furnished 1-bedroom, 720 sq.ft., AED 88,000/year, pet-friendly, shared pool, central AC, elevators.
    Availability: From December 1, 2025.
    Lease Terms: 12-month lease, AED 4,500 security deposit, maintenance by owner, sublease allowed, verified property.
    Landmarks: Near Dubai Marina Mall, Marina Walk, Jumeirah Beach.
    Transport: Close to DMCC Metro, bus stops, and tram.
    Next Steps:
    View the Property Card: Check the full listing on grok.com or x.com, using the property’s unique ID (a UUID from the id field).
    Contact the Agent: Reach out via the platform to discuss the lease or schedule a viewing.
    Prepare Documents: Have your ID, proof of income, and pet details ready, as it’s pet-friendly and allows subleasing.
    Confirm Terms: Verify payment terms (e.g., monthly or quarterly) and sign the lease agreement with the agent.
    Would you like me to check for similar pet-friendly properties in Dubai Marina as backup options, or are you set to proceed with Al Sahab Tower 2? I can also suggest questions to ask the agent if you need help.
    User: I’m good with this one. Thanks for the help! I’ll check the property card on grok.com.
    AI: You’re very welcome! I’m thrilled you found the perfect place in Al Sahab Tower 2. You can view the property card on grok.com to see all the details and connect with the agent. If you need any more assistance, like drafting questions for the agent or exploring other options, just let me know. Enjoy your new home in Dubai Marina, and happy leasing!
    Notes on the Conversation
    Schema Utilization: The AI uses fields like community (Dubai Marina), number_of_bedrooms (1), furnishing_status (FURNISHED), rent_charge (≤ AED 100,000), swimming_pool, pet_friendly, available_from, nearby_landmarks, public_transport_type, and media to filter and present relevant properties.
    Guiding Vague Queries: The AI starts by asking clarifying questions (city, budget, bedrooms) when the user is unsure, leveraging popular locations and common preferences.
    Refining Search: As the user provides more details (e.g., furnished, pool, pet-friendly), the AI narrows down options and provides specific listings with details like rent_charge, security_deposit, and maintenance_covered_by.
    Actionable Next Steps: The AI guides the user to platforms like grok.com for listings and provides practical advice on contacting agents or viewing properties.
    Natural Flow: The conversation feels human-like, with the AI adapting to the user’s preferences and offering relevant suggestions or clarifications.
    This interaction showcases how a RAG-based AI can use the Property entity to assist users in finding a lease property in the UAE, handling both vague and specific queries effectively. If you’d like a different scenario (e.g., a specific budget or location) or additional details, let me know!
    Most important is that this ai chatbot only support in UAE outside UAE he cannot answer because we have property detail of UAE only in database.

    LLM should only answer about the property what RAG has the context it should not answer outside the context. When users might greet the llm by hi hello good morning i that case llm should also greet with simple and nice message but source should be empty in that case no property detail should be shown. Some time users can sound like informative if they do not understand the word related the property for example what is property what is apartment what is holiday home ready rent type that time llm should answer by searching the google and give the simple and nice but source should be empty in that case no property detail should be shown. Sometime users can ask which is the best property in xyz location show the property which as premiumBoostingStatus, carouselBoostingStatus is active and bnb\*verification_status is verified first priorty should be set as 1. premiumBoostingStatus, carouselBoostingStatus active and bnb_verification_status is verified which as all 2. bnb_verification_status is verified 3.carouselBoostingStatus is active 4. premiumBoostingStatus is active. what if users can ask average price of in xyz location llm should check location and calcuate price and give the average value of property.
    \*\*\* that time also llm should not provide property detail in the source. \_\*\*
    Below are set of questions user can ask to get information.

    1. "Best Property" Queries
    These questions ask for the "best" properties in a given location, where "best" could be determined by factors like premiumBoostingStatus, carouselBoostingStatus, bnb_verification_status, amenities (e.g., swimming_pool, gym_fitness_center), furnishing_status, recent listing_date, or high-quality media. The AI can prioritize properties that are verified, boosted, or feature-rich.
    Sample Questions:
    "Show me the best 2-bedroom apartments in Dubai Marina for lease."
    "What are the top-rated villas in Palm Jumeirah based on amenities?"
    "Find the best-furnished property in Downtown Dubai with premium boosting."
    "List the top 3 verified properties in Jumeirah Beach Residence for rent."
    "Show me the best pet-friendly apartments in Business Bay with a balcony and pool."
    "What’s the best 1-bedroom condo in Abu Dhabi’s Al Reem Island with recent listings?"
    "Find the top property in Emirates Hills with a private pool and smart home features."
    How the AI Handles These:
    Retrieval: Filter properties by location (emirate, city, community, subcommunity) and prioritize based on premiumBoostingStatus (e.g., ACTIVE), bnb_verification_status (e.g., VERIFIED), or amenities (swimming_pool, gym_fitness_center).
    Ranking: Rank by quality indicators like boosting status, verification, or number of amenities. For example, a "best" property might be one with furnishing_status: FURNISHED, multiple media files, and ACTIVE status.
    Response: Provide a list or description of top properties, e.g., "The best property in Dubai Marina is a 2-bedroom furnished apartment in Cayan Tower, verified, with a private pool and premium boosting, available for AED 150,000/year."
    2. "Average Property Price" Queries
    These questions ask for the average lease price (rent_charge) of properties in a specific location, potentially filtered by attributes like number_of_bedrooms, furnishing_status, property_type, or lease_duration. The AI can aggregate rent_charge data to calculate averages.
    Sample Questions:
    "What’s the average lease price for 1-bedroom apartments in Dubai Marina?"
    "Show me the average rent for furnished villas in Palm Jumeirah."
    "What’s the average annual rent for 2-bedroom properties in Downtown Dubai?"
    "Find the average lease cost for unfurnished apartments in JLT."
    "What’s the average rental price for 3-bedroom houses in Abu Dhabi’s Al Raha Beach?"
    "Show me the average rent for short-term leases (6 months) in Business Bay."
    "What’s the average price for verified properties in Emirates Hills?"
    How the AI Handles These:
    Retrieval: Query properties by location and relevant filters (e.g., number_of_bedrooms, furnishing_status, rent_type).
    Aggregation: Calculate the mean of rent_charge for matching properties. For example, for "1-bedroom apartments in Dubai Marina," the AI sums rent_charge for all matching records and divides by the count.
    Response: Provide a clear answer, e.g., "The average lease price for 1-bedroom apartments in Dubai Marina is AED 95,000/year based on 50 listed properties."
    3. Other Analytical or Comparative Queries
    Users may also ask for comparisons, trends, or insights about properties in a location, leveraging fields like rent_charge, property_size, year_built, listing_date, or amenities. These questions go beyond "best" or "average" to explore patterns or options.
    Sample Questions:
    "Compare the lease prices of furnished vs. unfurnished apartments in Dubai Marina."
    "What’s the trend in rental prices for 2-bedroom properties in Downtown Dubai over the past year?"
    "Show me the most affordable 1-bedroom apartments in JBR compared to Business Bay."
    "What are the top amenities in properties listed in Palm Jumeirah?"
    "Find the newest properties (built after 2020) in Abu Dhabi’s Al Reem Island and their average rent."
    "What’s the price range for 3-bedroom villas in Emirates Hills with a private pool?"
    "Show me the number of verified vs. non-verified properties in JLT and their average rents."
    How the AI Handles These:
    Retrieval: Filter by location and relevant attributes (e.g., furnishing_status, year_built, bnb_verification_status).
    Analysis: Perform comparisons (e.g., furnished vs. unfurnished rent_charge), calculate ranges, or identify trends (e.g., group by listing_date to analyze price changes).
    Response: Provide a detailed comparison or insight, e.g., "Furnished apartments in Dubai Marina have an average rent of AED 110,000/year, while unfurnished ones average AED 85,000/year based on 75 listings."
    4. Vague or Guided Queries
    If users are unsure about their preferences, they might ask broad questions, and the AI can guide them using the schema’s attributes (e.g., suggesting popular locations, budgets, or amenities).
    Sample Questions:
    "What are the best areas in Dubai for leasing a family-friendly property?"
    "Help me find a good property to rent in Dubai, I’m not sure where to start."
    "What’s a reasonable budget for renting an apartment in Abu Dhabi?"
    "Show me popular properties in Dubai for a 12-month lease."
    "What are the best options for a first-time renter in Jumeirah?"
    "Find me a nice place to live in Dubai Marina, I don’t know much about rentals."
    "What are the top locations in Dubai for properties with beach access?"
    How the AI Handles These:
    Retrieval: Use popular locations (e.g., Dubai Marina, Downtown Dubai) or high-quality properties (premiumBoostingStatus: ACTIVE, bnb_verification_status: VERIFIED).
    Guidance: Ask follow-up questions like, “What’s your budget?” or “Do you need specific amenities like a pool or gym?” to refine the search.
    Response: Suggest options, e.g., “Dubai Marina is a popular area for rentals. The best properties there include furnished 1-bedroom apartments averaging AED 95,000/year with amenities like pools and gyms. Would you like to see specific listings?”
    Notes for Implementation
    "Best" Property Logic: The AI should define "best" using a scoring mechanism (e.g., weight premiumBoostingStatus, bnb_verification_status, number of amenities, or recent listing_date). For example, a property with ACTIVE boosting, VERIFIED status, and multiple amenities like swimming_pool and gym_fitness_center could rank higher.
    Price Aggregation: For average or range queries, the AI must query rent_charge and filter by relevant attributes (e.g., number_of_bedrooms, community). It should handle edge cases like missing data or outliers.
    Location Specificity: The AI should parse "XYZ location" to match emirate, city, community, or subcommunity (e.g., "XYZ" = Dubai Marina maps to community: Dubai Marina).
    Data Gaps: If no data is available for a specific query (e.g., no properties in a small subcommunity), the AI can suggest nearby locations or broader areas (e.g., "No listings in XYZ subcommunity, but here are properties in Dubai Marina.")
    Real-Time Data: If the system has access to recent listings (via listing_date or external sources like X posts or web searches), it can provide up-to-date insights.

    Most Important Feature i want to implement Gather users requirement. when user do not found desired property
    LLM should provide to option ir two ways
    1. ask alternate question 
    2. gather users requirement
    alternate question means if your was searching the property in sharjah but we do not have the property in sharjah then
    llm should check nearby location and suggest the user before suggesting it llm should check do that location exist in context or not. The alternate question it can about price number of bedrooms property type rent type propety size furnishing status or about aminities or parking it can be relate to property field only
    gathering means based on users requirement what were converstation happen between the ai and user it should summaries
    and send to admin by hitting the endpoint were we will collect the information which help our platform to gather the insights what user is looking for platform admin will give the requirment to the agency and agency will find the property in that location which help future users when new users search same property in that location or related price etc he will get it. before ending the chat llm should send the detail to xyz endpoint you may not save but when conversation is ended or user leave the conversation llm can send or it may also ask user do you want to save the requirement if users said yes or not llm should save the requirment by hitting the endpoint http://localhost:5000/backend/api/v1/user/requirement it will send to my main backend application
    Implementing the "Gather User Requirements" Feature
    Thank you for outlining this crucial feature! Gathering user requirements when no matching properties are found will enhance user satisfaction and provide valuable insights for your platform. I'll break this down into:
    Feature Logic Overview: A high-level explanation of how the LLM (me, as Grok) would implement this, based on the Property entity schema (e.g., fields like emirate, city, community, rent_charge, number_of_bedrooms, furnishing_status, property_type, amenities like swimming_pool or parking, etc.).
    Key Rules for Alternate Questions: Ensure suggestions are schema-aligned and verified.
    Key Rules for Gathering Requirements: Summarization and endpoint simulation.
    Sample Interactive Conversation: A demo where the user searches for a property in Sharjah (no matches), triggering the feature. This assumes a RAG system with no Sharjah listings in the database, and I simulate "checking context" (e.g., via schema fields or quick internal validation).
    This implementation treats the conversation as stateful, tracking user preferences across turns. For real deployment, integrate with your backend to hit the "xyz endpoint" (e.g., POST /admin/requirements with JSON payload like { "summary": "...", "user_id": "...", "preferences": {...} }).

    1. Feature Logic Overview
    Trigger: When a search yields 0 results (e.g., no properties match emirate: Sharjah in the database).
    Options Provided:
    Alternate Questions: LLM suggests 2-3 refined queries to broaden the search. These must relate to schema fields (e.g., nearby emirate like Dubai, adjust rent_charge, number_of_bedrooms, furnishing_status, property_type, rent_type, property_size, amenities like swimming_pool or parking).
    Before suggesting, "check context": Validate if the alternative exists in schema/database (e.g., does emirate: Dubai have listings? Use RAG query or internal flag).
    Gather Requirements: Summarize the conversation (key user prefs from history, like budget, location, amenities). Then, ask if user wants to "save" (send to admin). If yes/no, simulate sending to endpoint.
    When to Send to Endpoint:
    At conversation end (e.g., user says "bye" or leaves).
    Or proactively after gathering, if user confirms "save".
    Payload: JSON summary (e.g., { "user_query": "1BR furnished in Sharjah, AED 80k/year", "preferences": { "emirate": "Sharjah", "bedrooms": 1, "furnished": true }, "full_summary": "User sought pet-friendly apt but no matches..." }).
    Edge Cases:
    If user ignores options, gently re-prompt.
    Track conversation history for summarization (e.g., using session state).
    No moralizing; keep it helpful and concise.
    2. Key Rules for Alternate Questions
    Verification Step: Before suggesting, internally "check" if the alternative is viable (e.g., query database for existence of emirate: Dubai listings > 0). If not, skip or suggest further alternatives.
    Schema-Tied Suggestions: Only use entity fields:
    Location: Nearby emirates/cities (e.g., Sharjah → Dubai/Ajman).
    Price: Adjust rent_charge (e.g., +20% budget).
    Bedrooms/Bathrooms: ±1 to number_of_bedrooms/number_of_bathrooms.
    Type: Alternatives to property_type (e.g., apartment → villa).
    Rent Type: Vary rent_type (e.g., long-term → short-term).
    Size: Adjust property_size (e.g., +200 sq.ft.).
    Furnishing: Toggle furnishing_status (e.g., furnished → semi-furnished).
    Amenities/Parking: Add/remove like swimming_pool, pet_friendly, parking (JSON array).
    Phrasing: "Would you like to try searching for [alternative]?" to keep it engaging.
    3. Key Rules for Gathering Requirements
    Summarization: Extract from conversation history:
    User prefs: Location, budget, bedrooms, amenities, etc.
    Full thread: Key exchanges (e.g., "User asked for 1BR in Sharjah, budget AED 80k, pet-friendly").
    Insights: Gaps (e.g., "No Sharjah listings; user open to Dubai").
    User Consent: Always ask "Do you want to save this requirement so we can notify our team?" If yes, "send" to endpoint (simulate in response). If no, end gracefully.
    Endpoint Simulation: In this demo, I'll log it as "Sending to xyz endpoint..." with sample JSON. In production, use backend hook.
    4. Sample Interactive Conversation
    This demo assumes:
    No properties in emirate: Sharjah (0 results).
    Database has listings in nearby emirate: Dubai (verified via "context check").
    Conversation history tracked for summarization.
    Current date: September 19, 2025 (used for availability checks).
    User: Hi, I'm looking to lease a 1-bedroom furnished apartment in Sharjah, budget AED 80,000/year, pet-friendly with a pool.
    AI: I'd be happy to help you find a lease in Sharjah! Let me search for furnished 1-bedroom apartments with a pool, pet-friendly, under AED 80,000/year.
    After checking our listings... unfortunately, we don't have any matching properties in Sharjah right now. But I can help in two ways:
    Try alternate searches: Based on nearby areas and your preferences, I checked our database—Dubai has great options. For example:
    1-bedroom furnished apartments in Dubai (near Sharjah border, like Deira), same budget and features.
    Or, semi-furnished options in Sharjah if you're flexible on furnishing.
    Or, 2-bedroom apartments in Sharjah to give more space.
    Which alternate would you like to explore?
    Gather your requirements: I can summarize what you're looking for and share it with our team—they'll work with agencies to find or add matching properties soon.
    What would you prefer—alternate search or save your requirements?
    User: No luck in Sharjah? Okay, let's try the alternate in Dubai.
    AI: Got it, switching to Dubai (verified: we have 15+ matching listings there, close to Sharjah via quick drive). Searching for furnished 1-bedroom apartments with a pool, pet-friendly, under AED 80,000/year...
    Here are some options in Dubai (Deira area, ~20-min drive from Sharjah):
    Deira Tower: Furnished 1-bedroom, 650 sq.ft., shared pool, pet-friendly. Rent: AED 75,000/year. Available from October 15, 2025. Lease: 12 months, security deposit AED 4,000.
    Al Ghurair Residence: Furnished 1-bedroom, 700 sq.ft., pool, central AC. Rent: AED 78,000/year. Available from November 1, 2025. Pet-friendly, maintenance by owner.
    You can view the property cards for these. Does one catch your eye, or want to tweak (e.g., higher budget for more options)?
    User: Hmm, Deira Tower looks okay, but I'd prefer something closer to the Sharjah border. Also, can you check for 2 bedrooms instead?
    AI: Understood—focusing on Dubai areas near Sharjah (like Al Nahda or Al Qusais, ~10-15 min drive). Adjusting to 2-bedroom furnished apartments with a pool, pet-friendly, under AED 80,000/year...
    We have a couple of matches:
    Al Nahda Residences: Furnished 2-bedroom, 900 sq.ft., shared pool, pet-friendly. Rent: AED 76,000/year. Available from December 1, 2025. Lease: 12 months, sublease allowed.
    Qusais Metro Tower: Furnished 2-bedroom, 850 sq.ft., pool, elevators. Rent: AED 79,000/year. Available from November 20, 2025. Pet-friendly.
    Property cards available. If these still aren't quite right, we can try another alternate (e.g., villas instead of apartments) or gather your requirements to alert our team.
    User: Still not ideal. Let's gather my requirements instead—I want the team to find something in Sharjah.
    AI: No problem, I appreciate your patience! Let me summarize your requirements based on our chat so our platform can gather insights and task agencies to source properties like this.
    Requirements Summary:
    Location: Sharjah (preferred), open to nearby Dubai border areas like Deira/Al Nahda.
    Property Type: Apartment (1-2 bedrooms).
    Furnishing: Furnished.
    Budget: AED 80,000/year max.
    Key Amenities: Pet-friendly (small dog), shared/private pool.
    Other Prefs: 12-month lease, available by December 2025, central AC.
    Full Conversation Insight: You started with a 1BR in Sharjah but were flexible on bedrooms and nearby locations when no matches found. No interest in unfurnished or higher budget.
    Do you want to save this requirement? If yes, I'll send it to our admin team—they'll notify agencies to find or list matching properties, helping future searches like yours.
    User: Yes, please save it.
    AI: Perfect, saving your requirements now. Sending to the admin endpoint for review...
    Thanks for sharing—this will help us improve! If you'd like to explore more alternates now or end the chat, just let me know. Anything else?
    User: That's all for now, thanks!
    AI: You're welcome! If your requirements lead to new listings, we'll notify you. Have a great day—feel free to chat anytime.
    This sample demonstrates the feature in action: Trigger on no matches, offer options, verify alternates (e.g., Dubai exists), summarize accurately, and handle consent/endpoint. For production, hook the "send" to your actual API. If you need code snippets (e.g., for LLM prompt engineering) or more samples (e.g., different triggers), let me know!
    ## Endpoint will be http://localhost:5000/backend/api/v1/user/requirement

# Step 9 LLM Response : Send prompt to the LLM (gemini-2.5-flash-lite) to generate an answer.

# Step 10 Return Answer + Sources : Show the answer with citations from retrieved chunks.

# TECH STACK

1. FastApi language python ( @google/genai library for google gemini )
2. PostgreSQL ->
   DATABASE_URL=postgresql://postgres:1411ROHITT@localhost:5432/leasebnb
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=leasebnb
   DB_USER=postgres
   DB_PASSWORD=1411ROHITT
3. Embedding -> LLM_EMBED_MODEL_NAME=gemini-embedding-001
4. Vector ->
   VECTOR_DB_HOST=http://localhost:9200
   VECTOR_DB_USERNAME=elastic
   VECTOR_DB_PASSWORD=DkIedPPSCb
5. LLM -> LLM_EMBED_MODEL_NAME=gemini-embedding-001

For Every Step create the separate file

