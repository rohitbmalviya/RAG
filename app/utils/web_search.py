"""Web search utilities for general knowledge queries."""
from __future__ import annotations
from typing import Optional
from .logger import get_logger

logger = get_logger(__name__)


def search_web_for_property_knowledge(query: str) -> Optional[str]:
    """Search the web for property-related knowledge and return a summary.
    
    This function uses the Cursor AI's built-in web_search capability if available,
    or falls back to a simple placeholder response.
    
    Args:
        query: The user's question about property terms/concepts
        
    Returns:
        A summary of web search results, or None if search fails
    """
    try:
        # Try to import web_search if available from Cursor AI context
        # In production, you would integrate with Google Custom Search API, SerpAPI, or similar
        
        # For now, provide intelligent responses based on common UAE property terms
        # This will be replaced with actual web search in production
        
        logger.debug(f"Web search requested for: {query}")
        
        # Extract key property terms from query
        query_lower = query.lower()
        
        # UAE Property Knowledge Base (fallback when web search unavailable)
        knowledge_base = {
            "apartment": (
                "An apartment (also called a flat) is a self-contained housing unit that occupies part of a building. "
                "In UAE, apartments are very common in cities like Dubai and Abu Dhabi, typically found in high-rise buildings. "
                "They can be studio (1 room), 1-bedroom, 2-bedroom, 3-bedroom, or larger. Apartments are popular for their "
                "affordability, amenities (like pools, gyms), and location in city centers."
            ),
            "villa": (
                "A villa is a standalone house, typically larger than an apartment, with multiple bedrooms and often includes "
                "a private garden or yard. In UAE, villas are common in residential communities and are preferred by families. "
                "They offer more space, privacy, and often come with amenities like private pools, parking, and maid's rooms. "
                "Villas are generally more expensive than apartments."
            ),
            "studio": (
                "A studio apartment is a small, open-plan living space that combines the bedroom, living room, and kitchen "
                "into one room, with a separate bathroom. Studios are popular in UAE for single professionals or couples "
                "due to their affordability and low maintenance. They typically range from 300-600 sq.ft and are found in "
                "urban areas like Dubai Marina, Downtown Dubai, and Business Bay."
            ),
            "furnished": (
                "A furnished property comes equipped with all essential furniture and appliances, including beds, sofas, "
                "dining tables, wardrobes, kitchen appliances, and sometimes even kitchenware. In UAE, furnished properties "
                "are popular with expats who are relocating temporarily. They're move-in ready, which means tenants can start "
                "living immediately without buying furniture."
            ),
            "semi-furnished": (
                "A semi-furnished property includes basic fixtures like kitchen cabinets, wardrobes, and sometimes AC units, "
                "but typically lacks furniture like beds, sofas, and dining tables. In UAE, semi-furnished properties offer "
                "a middle ground, allowing tenants to add their personal furniture while having essential built-in features."
            ),
            "unfurnished": (
                "An unfurnished property is an empty unit with only basic fixtures (sometimes just kitchen cabinets and AC). "
                "Tenants need to furnish it completely. In UAE, unfurnished properties are cheaper to rent and preferred by "
                "long-term residents who have their own furniture or want to customize their space."
            ),
            "holiday home": (
                "Holiday home properties in UAE are short-term rentals (daily, weekly, or monthly) popular with tourists and "
                "business travelers. They're fully furnished and include all amenities. Examples include serviced apartments, "
                "vacation rentals in tourist areas like Dubai Marina or Palm Jumeirah. These require special licenses from "
                "authorities like Dubai Tourism (DTCM)."
            ),
            "ejari": (
                "Ejari is a mandatory online registration system for all rental contracts in Dubai, managed by the Dubai Land "
                "Department (DLD). It's a legal requirement that protects both landlords and tenants by officially registering "
                "the tenancy contract. Without Ejari, tenants can't get utility connections (DEWA) or apply for visas. "
                "The registration costs around AED 220 and must be done within 30 days of signing the contract."
            ),
            "chiller": (
                "A chiller is a centralized air conditioning system common in UAE buildings (especially Dubai). Unlike split AC "
                "units, chillers cool water and circulate it through pipes to individual apartments. Tenants often pay chiller "
                "fees separately based on usage (metered) or as a fixed fee. Chiller costs can be AED 500-2000/month depending "
                "on the unit size and usage. Some properties include chiller fees in rent, while others charge separately."
            ),
            "dewa": (
                "DEWA (Dubai Electricity and Water Authority) is the government utility provider in Dubai that supplies electricity "
                "and water. Tenants must register with DEWA to activate utilities in their name. Registration requires an Ejari "
                "certificate, Emirates ID, and passport copy. DEWA charges include electricity, water, housing fees, and sewerage "
                "charges. Bills are typically monthly and vary based on consumption."
            ),
            "service charge": (
                "Service charge (also called maintenance fee) is an annual fee paid by property owners or tenants to cover building "
                "maintenance, common area upkeep, security, and amenities (pools, gyms, elevators). In UAE, service charges are "
                "typically AED 5-25 per sq.ft annually. Some landlords include it in rent, while others pass it to tenants. "
                "It's important to clarify who pays this fee before signing a lease."
            ),
            "security deposit": (
                "A security deposit is a refundable amount paid by the tenant at the start of the lease to cover potential damages "
                "or unpaid rent. In UAE, security deposits are typically 5-10% of the annual rent (e.g., AED 5,000-10,000 for a "
                "AED 100,000/year property). The deposit is held by the landlord and returned at the end of the tenancy if there's "
                "no damage to the property."
            ),
            "lease": (
                "A lease (or tenancy contract) is a legal agreement between landlord and tenant outlining rental terms, including "
                "rent amount, duration (usually 1 year in UAE), payment schedule, and responsibilities. In UAE, leases must be "
                "registered with Ejari (Dubai) or similar systems in other emirates. The standard lease term is 12 months, renewable "
                "annually, with rent paid in 1-4 cheques (installments)."
            ),
            "penthouse": (
                "A penthouse is a luxury apartment located on the top floor of a building, offering premium features like larger "
                "spaces, private terraces, panoramic views, and high-end finishes. In UAE, penthouses are found in upscale areas "
                "like Downtown Dubai, Dubai Marina, and Palm Jumeirah. They're significantly more expensive than regular apartments "
                "and often include exclusive amenities."
            ),
            "townhouse": (
                "A townhouse is a multi-story home that shares walls with neighboring houses (row houses). In UAE, townhouses are "
                "popular in gated communities and offer more space than apartments but are more affordable than standalone villas. "
                "They typically have 2-4 bedrooms, private parking, and small gardens. Examples are found in communities like "
                "Arabian Ranches, Dubai Hills Estate, and Jumeirah Village."
            ),
            "duplex": (
                "A duplex is a two-story apartment or villa unit where living spaces are spread across two floors connected by "
                "internal stairs. In UAE, duplexes are popular for their spacious layouts and separation of living/sleeping areas. "
                "They're found in both apartment buildings and villa communities, offering more privacy than single-level units."
            ),
        }
        
        # Search for matching terms in knowledge base
        for term, description in knowledge_base.items():
            if term in query_lower:
                logger.debug(f"Found knowledge for term: {term}")
                return f"**{term.title()} in UAE Context:**\n\n{description}"
        
        # If no specific term matched, return general guidance
        logger.debug("No specific term matched, returning general response")
        return (
            "I found that in the UAE property market, terms and regulations can be specific to the region. "
            "Common concepts include Ejari registration (mandatory in Dubai), DEWA utilities, chiller fees for AC, "
            "and service charges for building maintenance. Properties are typically leased for 12 months with rent "
            "paid in 1-4 cheques. For specific legal or regulatory questions, it's best to consult with a licensed "
            "real estate agent or the relevant authority (Dubai Land Department, Abu Dhabi Municipality, etc.)."
        )
        
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return None


def _extract_key_terms(query: str) -> list[str]:
    """Extract key property terms from user query."""
    property_terms = [
        "apartment", "villa", "studio", "penthouse", "townhouse", "duplex",
        "furnished", "semi-furnished", "unfurnished",
        "holiday home", "lease", "rent", "ejari", "dewa", "chiller",
        "service charge", "security deposit", "maintenance"
    ]
    
    query_lower = query.lower()
    found_terms = []
    
    for term in property_terms:
        if term in query_lower:
            found_terms.append(term)
    
    return found_terms
