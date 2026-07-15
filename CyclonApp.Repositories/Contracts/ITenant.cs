using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CyclonApp.Database;

namespace CyclonApp.Repositories.Contracts
{
    public interface ITenant
    {
        int CurrentTenantId { get; }
        Task<Tenant?> GetCurrentTenantAsync();
    }
}
